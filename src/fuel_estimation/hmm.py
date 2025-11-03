import numpy as np
import pandas as pd
from typing import Optional

import numpy as np
import pandas as pd
from typing import Optional

def _logN(x, mu, sigma):
    """Gaussian log-pdf with numerical stability."""
    z = (x - mu) / (sigma + 1e-12)
    return -0.5*z*z - np.log(sigma + 1e-12) - 0.5*np.log(2*np.pi)

def _logsumexp(a, axis=None):
    """Numerically stable log-sum-exp."""
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)).squeeze(axis)

def hmm_refuel_probs_strict(
    work: pd.DataFrame,
    speed_col: str = "cmd_speed",
    stop_eps: float = 0.05,
    signal_col: Optional[str] = "fuel_est_kalman",
    max_gap: str = "15min",
    rate_smooth_win: int = 11,
    rate_clip_quant: tuple[float,float] = (0.5, 99.5),
    # MUCH STRICTER emissions: only high positive rates are refuel
    mu_C: float = -0.08,   sigma_C: float = 0.025,  # Consumption: negative rate
    mu_I: float =  0.00,   sigma_I: float = 0.015,  # Idle: ~zero rate
    mu_R: float =  0.35,   sigma_R: float = 0.15,   # Refuel: STRONG positive rate
    # Very stable transitions - avoid chatter
    A_base: np.ndarray = np.array([
        [0.998, 0.0015, 0.0005],   # C→C (very sticky)
        [0.003, 0.996,  0.001],    # I→I (very sticky)
        [0.300, 0.100,  0.600],    # R→R or exit quickly
    ]),
    pi: np.ndarray = np.array([0.96, 0.03, 0.01])
) -> pd.DataFrame:
    """
    STRICT HMM for refuel detection.
    Key changes:
    - Refuel state requires STRONG positive rate (mu_R=0.35, tight sigma)
    - Much tighter emission variances to reduce ambiguity
    - Stricter constraints on upward rate detection
    """
    
    w = work.copy()
    w["datetime"] = pd.to_datetime(w["datetime"])
    w = w.sort_values("datetime").reset_index(drop=True)

    # Select signal
    if signal_col and (signal_col in w.columns):
        sig = pd.to_numeric(w[signal_col], errors="coerce").astype(float)
    elif "fuel_est_kalman" in w.columns:
        sig = pd.to_numeric(w["fuel_est_kalman"], errors="coerce").astype(float)
    elif "fuel_est_rm" in w.columns:
        sig = pd.to_numeric(w["fuel_est_rm"], errors="coerce").astype(float)
    else:
        sig = pd.to_numeric(w["fuel"], errors="coerce").astype(float)

    t = w["datetime"].to_numpy(dtype="datetime64[ns]")
    dt = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
    dt[~np.isfinite(dt)] = 0.0
    max_gap_s = pd.to_timedelta(max_gap).total_seconds()

    # Compute rate
    sig_np = sig.to_numpy()
    valid_delta = (
        np.isfinite(sig_np[1:]) &
        np.isfinite(sig_np[:-1]) &
        (dt > 0) & (dt <= max_gap_s)
    )
    rate = np.zeros_like(sig_np, dtype=float)
    rate[1:][valid_delta] = (sig_np[1:][valid_delta] - sig_np[:-1][valid_delta]) / dt[valid_delta]

    # Smooth rate
    if rate_smooth_win and rate_smooth_win >= 3 and rate_smooth_win % 2 == 1:
        rate = pd.Series(rate).rolling(rate_smooth_win, center=True, min_periods=1).median().to_numpy()
    
    v = rate[np.isfinite(rate)]
    if v.size:
        lo, hi = np.nanpercentile(v, rate_clip_quant)
        rate = np.clip(rate, lo, hi)

    # Load speed
    speed = None
    if speed_col in w.columns:
        speed = pd.to_numeric(w[speed_col], errors="coerce").fillna(0).to_numpy()

    T = len(rate)
    C, I, R = 0, 1, 2

    # Emission matrix
    logB = np.column_stack([
        _logN(rate, mu_C, sigma_C),
        _logN(rate, mu_I, sigma_I),
        _logN(rate, mu_R, sigma_R)
    ])

    # STRICT CONSTRAINTS: refuel only when rate is STRONGLY positive
    # Detect sustained upward trends (3+ consecutive positive ticks)
    positive_rate = rate > 0.10  # Meaningful positive rate threshold
    up_streak = pd.Series(positive_rate).rolling(3, min_periods=3).sum().fillna(0).to_numpy() >= 3
    
    # Hard block refuel unless we have sustained upward movement
    logB[~up_streak, R] -= 50.0  # Nearly impossible
    
    # Block consumption and idle when rate is clearly positive (refueling)
    strong_up = rate > 0.15
    logB[strong_up, C] -= 50.0
    logB[strong_up, I] -= 50.0
    
    # Block refuel when rate is negative or near-zero
    not_refueling = rate < 0.08
    logB[not_refueling, R] -= 50.0

    # Glitch detection
    valid_rates = rate[np.isfinite(rate)]
    cap = np.nanpercentile(np.abs(valid_rates), 99.5) if valid_rates.size else np.inf
    glitch = np.abs(rate) > cap
    logB[glitch, :] = 0.0

    # No observations at big gaps
    no_obs = np.ones(T, dtype=bool)
    no_obs[1:] = ~valid_delta
    logB[no_obs, :] = 0.0

    # Time-varying transitions (speed-dependent)
    A = np.log(np.clip(A_base, 1e-12, 1.0))
    logA_t = np.tile(A[None, :, :], (T, 1, 1))
    
    if speed is not None:
        moving = np.abs(speed) > stop_eps
        # HARD suppress refuel while moving
        logB[moving, R] -= 100.0
        
        for i in np.where(moving)[0]:
            Arow = A_base.copy()
            Arow[:, R] = 1e-10  # Can't transition TO refuel
            for row_idx in range(3):
                if Arow[row_idx].sum() > 0:
                    Arow[row_idx] /= Arow[row_idx].sum()
            logA_t[i] = np.log(np.clip(Arow, 1e-12, 1.0))

    # Forward-backward algorithm
    log_pi = np.log(np.clip(pi / pi.sum(), 1e-12, 1.0))
    alpha = np.zeros((T, 3))
    beta = np.zeros((T, 3))
    
    alpha[0] = log_pi + logB[0]
    for i in range(1, T):
        alpha[i] = _logsumexp(alpha[i-1][:, None] + logA_t[i-1], axis=0) + logB[i]
    
    beta[-1] = 0.0
    for i in range(T-2, -1, -1):
        beta[i] = _logsumexp(logA_t[i] + (logB[i+1] + beta[i+1])[None, :], axis=1)

    log_gamma = alpha + beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)

    return pd.DataFrame({
        "datetime": w["datetime"],
        "rate": rate.astype(float),
        "p_consume": gamma[:, C],
        "p_idle": gamma[:, I],
        "p_refuel": gamma[:, R],
        "state_hat": np.array(["C", "I", "R"])[gamma.argmax(1)]
    })


def add_hmm_to_work_strict(work: pd.DataFrame, speed_col="cmd_speed", stop_eps=0.05) -> pd.DataFrame:
    """Add STRICT HMM refuel probabilities to work dataframe."""
    w = work.copy()
    w["datetime"] = pd.to_datetime(w["datetime"])
    w = w.sort_values("datetime").reset_index(drop=True)

    hmm_df = hmm_refuel_probs_strict(
        w,
        speed_col=speed_col,
        stop_eps=stop_eps,
        signal_col="fuel_est_kalman",
        max_gap="15min",
        rate_smooth_win=11,
        rate_clip_quant=(0.5, 99.5),
    )

    if len(hmm_df) == len(w):
        w["hmm_rate"] = pd.to_numeric(hmm_df["rate"], errors="coerce")
        w["hmm_p_refuel"] = pd.to_numeric(hmm_df["p_refuel"], errors="coerce")
    else:
        tmp = hmm_df.rename(columns={"rate": "hmm_rate", "p_refuel": "hmm_p_refuel"})
        w = w.merge(tmp[["datetime", "hmm_rate", "hmm_p_refuel"]], on="datetime", how="left")

    w["hmm_rate"] = pd.to_numeric(w["hmm_rate"], errors="coerce").fillna(0.0).astype(float)
    w["hmm_p_refuel"] = (
        pd.to_numeric(w["hmm_p_refuel"], errors="coerce")
        .fillna(0.0).clip(0.0, 1.0).astype(float)
    )

    print(f"STRICT HMM: {w['hmm_p_refuel'].notna().sum()} of {len(w)} points processed")
    return w


def hmm_events_from_probs_ultra_strict(
    work: pd.DataFrame,
    prob_col: str = "hmm_p_refuel",
    thr_on: float = 0.98,       # Very high threshold
    thr_off: float = 0.85,      # Strong hysteresis
    min_duration: str = "8min",  # Must be sustained
    merge_gap: str = "20min",
    estimate_col: Optional[str] = None,
    speed_col: str = "cmd_speed",
    stop_eps: float = 0.05,
    min_step: float = 50.0,     # Minimum fuel increase (liters)
    rate_col: str = "hmm_rate",  # NEW: use computed rate for validation
) -> pd.DataFrame:
    """
    ULTRA-STRICT event extraction from HMM probabilities.
    Only reports events with:
    - Very high probability peak (>0.98)
    - Sustained duration (8+ minutes)
    - Vehicle stopped throughout
    - Real fuel increase of 50+ liters
    """
    w = work.copy()
    w["datetime"] = pd.to_datetime(w["datetime"])
    w = w.sort_values("datetime").reset_index(drop=True)

    if prob_col not in w:
        return pd.DataFrame(columns=["datetime", "fuel_at_event", "step", "p_peak", "window_increase", "avg_rate"])

    t = w["datetime"].to_numpy()
    p = pd.to_numeric(w[prob_col], errors="coerce").fillna(0).to_numpy()

    # Hysteresis detection
    starts, ends = [], []
    inside = False
    s_idx = None
    for i, pi in enumerate(p):
        if not inside and pi >= thr_on:
            inside = True
            s_idx = i
        elif inside and pi <= thr_off:
            inside = False
            starts.append(s_idx)
            ends.append(i)
    if inside:
        starts.append(s_idx)
        ends.append(len(p))

    if not starts:
        return pd.DataFrame(columns=["datetime", "fuel_at_event", "step", "p_peak", "window_increase", "avg_rate"])

    # Filter by duration
    min_dt = pd.to_timedelta(min_duration).total_seconds()
    keep = []
    for s, e in zip(starts, ends):
        dur = (t[e-1] - t[s]).astype("timedelta64[s]").astype(float)
        if dur >= min_dt:
            keep.append((s, e))

    if not keep:
        return pd.DataFrame(columns=["datetime", "fuel_at_event", "step", "p_peak", "window_increase", "avg_rate"])

    # Choose estimate signal
    if estimate_col and estimate_col in w:
        s_est = pd.to_numeric(w[estimate_col], errors="coerce").astype(float).ffill().bfill()
    elif "fuel_est_kalman_2D" in w:
        s_est = pd.to_numeric(w["fuel_est_kalman_2D"], errors="coerce").astype(float).ffill().bfill()
    elif "fuel_est_kalman" in w:
        s_est = pd.to_numeric(w["fuel_est_kalman"], errors="coerce").astype(float).ffill().bfill()
    else:
        s_est = pd.to_numeric(w["fuel"], errors="coerce").astype(float).ffill().bfill()

    s_est_idx = pd.Series(s_est.values, index=w["datetime"])
    pre_med_series = s_est_idx.rolling("10min", min_periods=3).median()
    post_med_series = s_est_idx.iloc[::-1].rolling("10min", min_periods=3).median().iloc[::-1]

    # Speed gate
    is_moving = None
    if speed_col in w.columns:
        spd = pd.to_numeric(w[speed_col], errors="coerce").fillna(0)
        is_moving = (spd.abs() > stop_eps).to_numpy()

    rows = []
    for s, e in keep:
        seg_p = p[s:e]
        peak_rel = int(np.argmax(seg_p))
        i_evt = s + peak_rel
        t_evt = pd.to_datetime(t[i_evt])

        # Gate 0: Check broader context - fuel must be INCREASING in wider window
        # Look at ±15min around the event
        context_win = pd.Timedelta("15min")
        context_mask = (w["datetime"] >= (t_evt - context_win)) & (w["datetime"] <= (t_evt + context_win))
        context_fuel = pd.to_numeric(w.loc[context_mask, estimate_col if estimate_col and estimate_col in w.columns else "fuel"], errors="coerce")
        
        if len(context_fuel) > 10:
            # Linear regression to detect overall trend
            context_times = (w.loc[context_mask, "datetime"] - t_evt).dt.total_seconds().values
            valid_idx = np.isfinite(context_fuel.values) & np.isfinite(context_times)
            if np.sum(valid_idx) > 5:
                slope = np.polyfit(context_times[valid_idx], context_fuel.values[valid_idx], 1)[0]
                # Slope must be positive (fuel increasing over time)
                if slope < 0.05:  # Less than 0.05 L/s increase = not refueling
                    continue

        # Gate 1: Vehicle stopped throughout the event window
        if is_moving is not None:
            if np.any(is_moving[s:e]):
                continue

        # Gate 2: RATE validation - must have positive rates during the event
        if rate_col in w.columns:
            rate_seg = pd.to_numeric(w[rate_col].iloc[s:e], errors="coerce").to_numpy()
            # Check: at least 60% of the segment has positive rate (increased from 50%)
            positive_frac = np.sum(rate_seg > 0.10) / len(rate_seg)
            if positive_frac < 0.60:
                continue
            
            # Check: average rate during segment is meaningfully positive
            avg_rate = np.nanmean(rate_seg)
            if avg_rate < 0.18:  # Increased from 0.15
                continue
            
            # NEW: No negative rates allowed (strict)
            if np.any(rate_seg < -0.05):
                continue  # Any significant consumption = not refuel

        # Gate 3: Real fuel step increase
        pre_med = float(pre_med_series.loc[t_evt]) if t_evt in pre_med_series.index else np.nan
        post_med = float(post_med_series.loc[t_evt]) if t_evt in post_med_series.index else np.nan
        
        if not (np.isfinite(pre_med) and np.isfinite(post_med)):
            continue
        
        step = post_med - pre_med
        if step < min_step:  # Must be meaningful increase
            continue
        
        # Gate 4: Check that fuel actually INCREASED during the window (not just noisy)
        fuel_start = float(s_est.iloc[s])
        fuel_end = float(s_est.iloc[e-1])
        window_increase = fuel_end - fuel_start
        if window_increase < 40.0:  # Increased from 30L
            continue
        
        # Gate 5: Monotonicity check - fuel should mostly increase throughout segment
        seg_fuel = s_est.iloc[s:e].values
        if len(seg_fuel) > 3:
            diffs = np.diff(seg_fuel)
            increasing_frac = np.sum(diffs > 0) / len(diffs)
            if increasing_frac < 0.55:  # At least 55% of steps should be increasing
                continue

        rows.append({
            "datetime": t_evt,
            "fuel_at_event": float(w["fuel"].iat[i_evt]) if "fuel" in w else np.nan,
            "step": step,
            "p_peak": float(seg_p.max()),
            "window_increase": window_increase,
            "avg_rate": avg_rate if rate_col in w.columns else np.nan,
        })

    if not rows:
        return pd.DataFrame(columns=["datetime", "fuel_at_event", "step", "p_peak", "window_increase", "avg_rate"])

    ev = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    # Merge nearby events
    if len(ev) > 1:
        mgap = pd.to_timedelta(merge_gap)
        merged = [ev.iloc[0].to_dict()]
        for _, r in ev.iloc[1:].iterrows():
            if (r["datetime"] - merged[-1]["datetime"]) <= mgap:
                if r["step"] > merged[-1]["step"]:
                    merged[-1] = r.to_dict()
            else:
                merged.append(r.to_dict())
        ev = pd.DataFrame(merged)

    return ev.reset_index(drop=True)


def plot_fuel_with_hmm(work: pd.DataFrame, events: Optional[pd.DataFrame] = None, 
                       title: str = "Fuel + HMM Refuel Detection") -> 'go.Figure':
    """
    Plot fuel levels with HMM refuel probability overlay and detected events.
    
    Args:
        work: DataFrame with 'datetime', 'fuel', 'fuel_est_*', and 'hmm_p_refuel'
        events: Optional DataFrame with detected refuel events
        title: Plot title
    
    Returns:
        Plotly figure
    """
    import plotly.graph_objects as go
    
    w = work.copy()
    w["datetime"] = pd.to_datetime(w["datetime"])
    t = w["datetime"]
    y = pd.to_numeric(w["fuel"], errors="coerce").astype(float)
    p = pd.to_numeric(w.get("hmm_p_refuel", pd.Series(index=w.index, dtype=float)), 
                      errors="coerce").fillna(0)

    fig = go.Figure()
    
    # Raw fuel
    fig.add_trace(go.Scatter(x=t, y=y, name="Fuel (raw)", mode="lines",
                             line=dict(color="blue")))

    # Fuel estimates (optional)
    if "fuel_est_rm" in w.columns:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_est_rm"], name="RM estimate",
                                mode="lines", line=dict(dash="dot", color="orange")))
    if "fuel_est_kalman" in w.columns:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_est_kalman"], name="Kalman 1D",
                                mode="lines", line=dict(dash="dash", color="red")))
    if "fuel_est_kalman_2D" in w.columns:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_est_kalman_2D"], name="Kalman 2D",
                                mode="lines", line=dict(dash="longdash", color="purple")))

    # HMM probability on secondary y-axis
    fig.add_trace(go.Scatter(x=t, y=p, name="P(refuel)", mode="lines",
                            yaxis="y2", fill="tozeroy", opacity=0.35,
                            line=dict(color="rgba(0,0,0,0.3)")))

    # Mark detected events
    if events is not None and not events.empty:
        e = events.copy()
        e["datetime"] = pd.to_datetime(e["datetime"])
        y_evt = np.interp(e["datetime"].astype("int64"), t.astype("int64"), y.values)
        fig.add_trace(go.Scatter(x=e["datetime"], y=y_evt, mode="markers",
                                marker=dict(color="#1a9850", size=10, symbol="diamond"),
                                name="Refuels (detected)"))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Time", rangeslider=dict(visible=True)),
        yaxis=dict(title="Fuel", side="left"),
        yaxis2=dict(title="P(refuel)", overlaying="y", side="right", range=[0, 1])
    )

    return fig
