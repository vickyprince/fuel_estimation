import numpy as np
import pandas as pd
from typing import Optional

from typing import Optional, Tuple, Set

def _normalize_cand_mode(mode: Optional[str]) -> str:
    """Return one of: 'raw', 'est', 'both_or', 'both_and'."""
    if mode is None:
        return "both_or"
    m = mode.strip().lower()
    if m in ("raw", "est"):
        return m
    if m in ("both_and", "and"):
        return "both_and"
    if m in ("both", "both_or", "or"):
        return "both_or"
    return "both_or"

def _combine_indices(idx_raw: Set[int], idx_est: Set[int], mode: str) -> list[int]:
    if mode == "raw":
        return sorted(idx_raw)
    if mode == "est":
        return sorted(idx_est)
    if mode == "both_and":
        return sorted(idx_raw.intersection(idx_est))
    # default both_or
    return sorted(idx_raw.union(idx_est))


def _robust_sigma_resid(x: pd.Series, win: int = 5) -> float:
    s = pd.Series(x).astype(float)
    sm = s.rolling(win, center=True, min_periods=1).median()
    resid = (s - sm).dropna()
    if resid.empty:
        return 1e-6
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(max(1.4826 * mad, 1e-6))

def _prep_series(work: pd.DataFrame, window_pts: int):
    """Return (w_sorted, s_est_idx, s_raw, d_est, d_raw)."""
    w = work.copy()
    w["datetime"] = pd.to_datetime(w["datetime"])
    w = w.sort_values("datetime").reset_index(drop=True)

    # choose estimate series exactly like detector uses
    if "fuel_est_kalman_2D" in w:
        s_est = w["fuel_est_kalman_2D"].astype(float)
    elif "fuel_est_kalman" in w:
    # if "fuel_est_kalman" in w:
        s_est = w["fuel_est_kalman"].astype(float)
    elif "fuel_est_rm" in w:
        s_est = w["fuel_est_rm"].astype(float)
    else:
        s_est = w["fuel"].astype(float).rolling(
            window_pts, center=True, min_periods=max(3, window_pts // 3)
        ).median()
    s_est = s_est.bfill().ffill()
    s_est_idx = pd.Series(s_est.values, index=w["datetime"])

    # raw path used for candidates + anti-rebound
    s_raw = w["fuel"].astype(float).rolling(
        max(5, (window_pts // 2) | 1), center=True, min_periods=3
    ).median().bfill().ffill()

    d_raw = s_raw.diff()
    d_est = s_est.diff()
    return w, s_est_idx, s_raw, d_est, d_raw

def _candidate_sets(d_raw: pd.Series, d_est: pd.Series, q: float) -> Tuple[Set[int], Set[int], float, float]:
    """Return (idx_raw, idx_est, thr_raw, thr_est)."""
    thr_raw = np.inf
    thr_est = np.inf
    pos_raw = d_raw[d_raw > 0].dropna()
    pos_est = d_est[d_est > 0].dropna()
    if not pos_raw.empty:
        thr_raw = float(pos_raw.quantile(q))
    if not pos_est.empty:
        thr_est = float(pos_est.quantile(q))
    idx_raw = set(d_raw.index[d_raw >= thr_raw].tolist()) if np.isfinite(thr_raw) else set()
    idx_est = set(d_est.index[d_est >= thr_est].tolist()) if np.isfinite(thr_est) else set()
    return idx_raw, idx_est, thr_raw, thr_est

def _pre_post_medians(s_est_idx: pd.Series, pre: str, post: str):
    """
    Compute rolling medians in the same way the detector does:
    pre_med(t)  = rolling(pre).median() at t
    post_med(t) = reverse( rolling(post).median() ) at t
    """
    pre_med_series  = s_est_idx.rolling(pre,  min_periods=3).median()
    post_med_series = s_est_idx.iloc[::-1].rolling(post, min_periods=3).median().iloc[::-1]
    return pre_med_series, post_med_series

def _stop_fraction(
    w_sorted: pd.DataFrame,
    t,
    speed_col: str,
    stop_eps: float,
    stop_pre: str,
    stop_post: str
) -> float:
    """Return fraction of |speed| <= stop_eps in [t-stop_pre, t+stop_post]. NaN if no samples."""
    if speed_col not in w_sorted.columns:
        return np.nan
    spd = pd.to_numeric(w_sorted[speed_col], errors="coerce")
    spd = spd.rolling(3, center=True, min_periods=1).median().ffill().bfill()
    m = (w_sorted["datetime"] >= t - pd.Timedelta(stop_pre)) & (w_sorted["datetime"] <= t + pd.Timedelta(stop_post))
    seg = spd[m].dropna()
    if len(seg) == 0:
        return np.nan
    return float((seg.abs() <= stop_eps).mean())


def _sustain_stats(s_est_idx, t, pre_med, min_step,
                   sustain_post="15min", level_k=0.7):
    """Return (level, frac_above). If we can't evaluate, return (NaN, NaN)."""
    m_post = (s_est_idx.index > t) & (s_est_idx.index <= t + pd.Timedelta(sustain_post))
    post_vals = s_est_idx.loc[m_post]
    if (not np.isfinite(pre_med)) or post_vals.empty:
        return np.nan, np.nan  # signal 'skip'
    level = float(pre_med + level_k * min_step)
    frac  = float((post_vals >= level).mean())
    return level, frac

def _robust_pre_post_at(s_est_idx, t, pre, post, pre_med_series, post_med_series):
    pre_med = float(pre_med_series.loc[t])
    if not np.isfinite(pre_med):
        m_pre = (s_est_idx.index >= t - pd.Timedelta(pre)) & (s_est_idx.index < t)
        pre_vals = s_est_idx.loc[m_pre]
        if len(pre_vals) >= 1:
            pre_med = float(pre_vals.median())
        else:
            # last value before t (if any)
            pre_med = float(s_est_idx.loc[:t].iloc[-1]) if len(s_est_idx.loc[:t]) else np.nan

    post_med = float(post_med_series.loc[t])
    if not np.isfinite(post_med):
        m_post = (s_est_idx.index > t) & (s_est_idx.index <= t + pd.Timedelta(post))
        post_vals = s_est_idx.loc[m_post]
        if len(post_vals) >= 1:
            post_med = float(post_vals.median())
        else:
            # first value after t (if any)
            post_med = float(s_est_idx.loc[t:].iloc[0]) if len(s_est_idx.loc[t:]) else np.nan

    return pre_med, post_med

def detect_refuel_events(
    work: pd.DataFrame,
    q: float = 0.98,
    window_pts: int = 11,
    pre: str = "10min",
    post: str = "10min",
    z: float = 2.7,
    cand_source="both",
    min_frac: float = 0.03,
    min_gap_seconds: int = 20*60,
    anti_glitch_window: str = "3min",
    anti_glitch_ratio: float = 0.7,
    speed_col: str = "cmd_speed",
    stop_eps: float = 0.05,
    stop_frac: float = 0.80,
    stop_pre: str = "6min",
    stop_post: str = "3min",
    require_stop: bool = True,
    sustain_post="5min",      # how long to check after t
    sustain_level_k=0.7,       # level = pre_med + k * min_step
    sustain_frac_req=0.60,     # need ≥60% of points above that level
    step_multiplier=1.15,      # require step ≥ 1.15×min_step (margin)
) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame(columns=["datetime","delta","fuel_at_event","step","stop_frac_obs"])

    # prep series (shared)
    w, s_est_idx, s_raw, d_est, d_raw = _prep_series(work, window_pts)

    mode = _normalize_cand_mode(cand_source)

    idx_raw, idx_est, thr_raw, thr_est = _candidate_sets(d_raw, d_est, q)
    cand_idx = _combine_indices(idx_raw, idx_est, mode)
    if not cand_idx:
        return pd.DataFrame(columns=["datetime","delta","fuel_at_event","step","stop_frac_obs"])

    # thresholds for step test (shared)
    sigma_est = _robust_sigma_resid(s_est_idx)
    rng_est   = float(s_est_idx.max() - s_est_idx.min())
    min_step  = max(z * sigma_est, min_frac * rng_est)

    pre_med_series, post_med_series = _pre_post_medians(s_est_idx, pre, post)

    out = []
    tvals = w["datetime"]
    for i in cand_idx:
        t = tvals.iat[i]

        raw_at = float(d_raw.iat[i]) if np.isfinite(d_raw.iat[i]) else np.nan
        est_at = float(d_est.iat[i]) if np.isfinite(d_est.iat[i]) else np.nan

        cand_raw = np.isfinite(raw_at) and (raw_at >= thr_raw)
        cand_est = np.isfinite(est_at) and (est_at >= thr_est)

        if mode == "raw":
            spike_ok = cand_raw
        elif mode == "est":
            spike_ok = cand_est
        elif mode == "both_and":
            spike_ok = cand_raw and cand_est
        else:  # both_or
            spike_ok = cand_raw or cand_est

        if not spike_ok:
            continue

        # anti-rebound (same)
        m_recent = (tvals >= t - pd.Timedelta(anti_glitch_window)) & (tvals < t)
        neg_recent = d_raw[m_recent].min() if m_recent.any() else np.nan
        raw_at = float(d_raw.iat[i]) if np.isfinite(d_raw.iat[i]) else 0.0
        anti_blocks = (np.isfinite(neg_recent) and (-neg_recent) >= anti_glitch_ratio * max(raw_at, 0.0))
        if anti_blocks:
            continue

        # step test (same)
        pre_med, post_med = _robust_pre_post_at(s_est_idx, t, pre, post, pre_med_series, post_med_series)
        step = post_med - pre_med
        if not np.isfinite(step) or step < step_multiplier * min_step:
            continue

        # --- sustain test (new) ---
        lvl, frac_above = _sustain_stats(
            s_est_idx, t, pre_med, min_step,
            sustain_post=sustain_post, level_k=sustain_level_k
        )
        if not np.isfinite(lvl) or frac_above < sustain_frac_req:
            continue

        # stop gate (same)
        stop_frac_obs = np.nan
        if require_stop and (speed_col in w.columns):
            stop_frac_obs = _stop_fraction(w, t, speed_col, stop_eps, stop_pre, stop_post)
            if not np.isfinite(stop_frac_obs) or (stop_frac_obs < stop_frac):
                continue

        out.append({
            "datetime": t,
            "delta": float(d_est.iat[i]) if np.isfinite(d_est.iat[i]) else 0.0,
            "fuel_at_event": float(w["fuel"].iat[i]),
            "step": step,
            "stop_frac_obs": stop_frac_obs
        })

    if not out:
        return pd.DataFrame(columns=["datetime","delta","fuel_at_event","step","stop_frac_obs"])

    ev = pd.DataFrame(out).sort_values("datetime").reset_index(drop=True)

    # merge by time gap, keep largest step (same)
    if len(ev) > 1:
        merged = [ev.iloc[0].to_dict()]
        for _, r in ev.iloc[1:].iterrows():
            if (r["datetime"] - merged[-1]["datetime"]).total_seconds() <= min_gap_seconds:
                if r["step"] > merged[-1]["step"]:
                    merged[-1] = r.to_dict()
            else:
                merged.append(r.to_dict())
        ev = pd.DataFrame(merged)

    return ev.reset_index(drop=True)

# ----------------------- explainer (exactly same gates) -----------------------

def explain_refuel_at(
    work: pd.DataFrame, when,
    q=0.98, window_pts=11, pre="10min", post="10min",
    z=2.7, min_frac=0.03, cand_source="both",
    anti_glitch_window="3min", anti_glitch_ratio=0.9,   # match detector
    speed_col="cmd_speed", stop_eps=0.05, stop_frac=0.80,
    stop_pre="6min", stop_post="3min",
    require_stop=True,
    # NEW – match detector:
    sustain_post="15min", sustain_level_k=0.7, sustain_frac_req=0.60,
    step_multiplier=1.15
) -> pd.Series:
    """Same gates as the detector. Returns all intermediate quantities."""
    w, s_est_idx, s_raw, d_est, d_raw = _prep_series(work, window_pts)

    t_req = pd.to_datetime(when)
    if not (w["datetime"].min() <= t_req <= w["datetime"].max()):
        raise ValueError("`when` is outside data range")

    mode = _normalize_cand_mode(cand_source)

    idx_raw, idx_est, thr_raw, thr_est = _candidate_sets(d_raw, d_est, q)

    # snap to nearest sample (must match detector)
    idx = int((w["datetime"] - t_req).abs().idxmin())
    t   = w["datetime"].iat[idx]

    d_raw_at = float(d_raw.iat[idx]) if np.isfinite(d_raw.iat[idx]) else np.nan
    d_est_at = float(d_est.iat[idx]) if np.isfinite(d_est.iat[idx]) else np.nan
    cand_raw = np.isfinite(d_raw_at) and (d_raw_at >= thr_raw)
    cand_est = np.isfinite(d_est_at) and (d_est_at >= thr_est)

    # mode-combined spike
    if mode == "raw":
        cand_pass = cand_raw;  spike_label = "Spike RAW (raw only)"
    elif mode == "est":
        cand_pass = cand_est;  spike_label = "Spike EST (est only)"
    elif mode == "both_and":
        cand_pass = cand_raw and cand_est; spike_label = "Spike BOTH (raw AND est)"
    else:
        cand_pass = cand_raw or cand_est;  spike_label = "Spike ANY (raw OR est)"

    # anti-glitch (same as detector)
    tvals = w["datetime"]
    m_recent = (tvals >= t - pd.Timedelta(anti_glitch_window)) & (tvals < t)
    neg_recent = d_raw[m_recent].min() if m_recent.any() else np.nan
    anti_blocks = (np.isfinite(neg_recent) and
                   (-neg_recent) >= anti_glitch_ratio * max(d_raw_at if np.isfinite(d_raw_at) else 0.0, 0.0))

    # step (with multiplier; skip if NaN at edges)
    pre_med_series, post_med_series = _pre_post_medians(s_est_idx, pre, post)
    pre_med, post_med = _robust_pre_post_at(s_est_idx, t, pre, post, pre_med_series, post_med_series)
    step = post_med - pre_med
    min_step = max(z * _robust_sigma_resid(s_est_idx), min_frac * float(s_est_idx.max() - s_est_idx.min()))
    step_threshold = step_multiplier * min_step
    if np.isnan(step):
        step_ok = True; step_note = "skipped (NaN; edge/sparse)"
    else:
        step_ok = (step >= step_threshold); step_note = ""

    # sustain (NEW – same as detector)
    sustain_level, sustain_frac = _sustain_stats(
        s_est_idx, t, pre_med, min_step,
        sustain_post=sustain_post, level_k=sustain_level_k
    )
    sustain_ok = (np.isfinite(sustain_level) and sustain_frac >= sustain_frac_req)

    # stop gate (same)
    stop_frac_obs = np.nan; stop_ok = True
    if require_stop and (speed_col in w.columns):
        stop_frac_obs = _stop_fraction(w, t, speed_col, stop_eps, stop_pre, stop_post)
        stop_ok = (np.isfinite(stop_frac_obs) and (stop_frac_obs >= stop_frac))

    # final decision (identical to detector)
    decision = (cand_pass and (not anti_blocks) and step_ok and sustain_ok and stop_ok)

    return pd.Series(dict(
        when=pd.to_datetime(t),
        idx=idx,
        # spikes
        d_raw_at=d_raw_at, thr_raw=thr_raw, cand_raw=cand_raw,
        d_est_at=d_est_at, thr_est=thr_est, cand_est=cand_est,
        cand_any=bool(cand_raw or cand_est),
        cand_pass=bool(cand_pass),
        cand_source=cand_source, cand_mode=mode, spike_label=spike_label,
        # anti-glitch
        anti_recent_min=neg_recent, anti_ratio_req=anti_glitch_ratio, anti_blocks=bool(anti_blocks),
        # step
        pre_med=pre_med, post_med=post_med, step=step,
        min_step=min_step, step_multiplier=step_multiplier,
        step_threshold=step_threshold, step_ok=bool(step_ok), step_note=step_note,
        # sustain
        sustain_post=sustain_post, sustain_level_k=sustain_level_k, sustain_frac_req=sustain_frac_req,
        sustain_level=sustain_level, sustain_frac=sustain_frac, sustain_ok=bool(sustain_ok),
        # stop
        stop_frac_obs=stop_frac_obs, stop_ok=bool(stop_ok),
        # final
        decision=bool(decision)
    ))


def _fmt_num(x):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):  return "NaN"
        if np.isinf(x):  return "∞" if x > 0 else "-∞"
        return f"{x:,.3f}"
    if isinstance(x, (int, np.integer)):
        return f"{x:,d}"
    return str(x)

def explain_refuel_table(
    work: pd.DataFrame, when,
    q=0.98, window_pts=11, pre="10min", post="10min",
    z=2.7, min_frac=0.03, cand_source="both",
    anti_glitch_window="3min", anti_glitch_ratio=0.7,
    speed_col="cmd_speed", stop_eps=0.05, stop_frac=0.80,
    stop_pre="6min", stop_post="3min",
    require_stop=True,
    pretty_print=True
) -> pd.DataFrame:
    s = explain_refuel_at(
        work, when,
        q=q, window_pts=window_pts, pre=pre, post=post,
        z=z, min_frac=min_frac, cand_source=cand_source,
        anti_glitch_window=anti_glitch_window, anti_glitch_ratio=anti_glitch_ratio,
        speed_col=speed_col, stop_eps=stop_eps, stop_frac=stop_frac,
        stop_pre=stop_pre, stop_post=stop_post, require_stop=require_stop
    )

    rows = []

    # Individual spikes (always show)
    rows.append({"Check": "Spike (raw Δ ≥ thr_raw)",
                 "Ideal/Threshold": f"Δ ≥ {s['thr_raw']}",
                 "Actual": s["d_raw_at"], "Pass": bool(s["cand_raw"]), "Details": ""})
    rows.append({"Check": "Spike (est Δ ≥ thr_est)",
                 "Ideal/Threshold": f"Δ ≥ {s['thr_est']}",
                 "Actual": s["d_est_at"], "Pass": bool(s["cand_est"]), "Details": ""})

    # Combined spike row (matches mode)
    mode = s["cand_mode"]
    if mode == "raw":
        spike_label = "Spike RAW (Δ_raw ≥ thr_raw)"
        spike_ideal = f"Δ_raw ≥ {s['thr_raw']}"
        spike_actual = s["d_raw_at"]
    elif mode == "est":
        spike_label = "Spike EST (Δ_est ≥ thr_est)"
        spike_ideal = f"Δ_est ≥ {s['thr_est']}"
        spike_actual = s["d_est_at"]
    elif mode == "both_and":
        spike_label = "Spike BOTH (raw AND est)"
        spike_ideal = f"(Δ_raw ≥ {s['thr_raw']}) AND (Δ_est ≥ {s['thr_est']})"
        spike_actual = f"{_fmt_num(s['d_raw_at'])} AND {_fmt_num(s['d_est_at'])}"
    else:  # both_or
        spike_label = "Spike ANY (raw OR est)"
        spike_ideal = f"(Δ_raw ≥ {s['thr_raw']}) OR (Δ_est ≥ {s['thr_est']})"
        spike_actual = f"{_fmt_num(s['d_raw_at'])} OR {_fmt_num(s['d_est_at'])}"

    rows.append({"Check": spike_label,
                 "Ideal/Threshold": spike_ideal,
                 "Actual": spike_actual,
                 "Pass": bool(s["cand_pass"]), "Details": ""})

    # Anti-glitch
    rows.append({"Check": "Anti-glitch (no dip-then-rebound)",
                 "Ideal/Threshold": f"(-min recent Δ) < {s['anti_ratio_req']} × max(Δ_raw, 0)",
                 "Actual": f"{_fmt_num(-(s['anti_recent_min'] if pd.notna(s['anti_recent_min']) else np.nan))} vs Δ_raw={_fmt_num(s['d_raw_at'])}",
                 "Pass": (not bool(s["anti_blocks"])),
                 "Details": "Blocks when prior negative dip is large vs current rise"})

    # Step (skip gate if NaN)
    rows.append({
        "Check": "Step (post_med - pre_med ≥ step_threshold)",
        "Ideal/Threshold": f"≥ {s['step_threshold']}",
        "Actual": s["step"],
        "Pass": bool(s["step_ok"]),
        "Details": (f"pre_med={_fmt_num(s['pre_med'])}, post_med={_fmt_num(s['post_med'])}"
                    + (f" | {s['step_note']}" if s.get("step_note") else ""))
    })

    # Sustain (NEW)
    rows.append({
        "Check": f"Sustain ≥{float(s['sustain_frac_req']):.0%} for {s['sustain_post']}",
        "Ideal/Threshold": f"post ≥ pre + {float(s['sustain_level_k']):.2f}×min_step (level={_fmt_num(s['sustain_level'])})",
        "Actual": f"frac_above={float(s['sustain_frac']):.2%}",
        "Pass": bool(s['sustain_ok']),
        "Details": s.get("sustain_note","")
    })

    # Stop gate
    if require_stop and (speed_col in work.columns):
        has_obs = not pd.isna(s["stop_frac_obs"])
        rows.append({"Check": f"Stopped fraction (|speed|≤{stop_eps})",
                     "Ideal/Threshold": f"≥ {stop_frac}",
                     "Actual": (s["stop_frac_obs"] if has_obs else "no data"),
                     "Pass": bool(s["stop_ok"]) if has_obs else False,
                     "Details": f"Window: {stop_pre}..+{stop_post}"})
    else:
        rows.append({"Check": "Stopped fraction", "Ideal/Threshold": "n/a",
                     "Actual": "n/a", "Pass": True, "Details": "Stop gate not required"})

    # Final decision + reason
    failed = []
    if not bool(s["cand_pass"]): failed.append("no spike")
    if bool(s["anti_blocks"]):   failed.append("anti-glitch")
    if not bool(s["step_ok"]):   failed.append("step<threshold")
    if not bool(s["sustain_ok"]):failed.append("no sustain")
    if require_stop and (speed_col in work.columns) and (not bool(s["stop_ok"])): failed.append("not stopped")
    reason = "PASS" if bool(s["decision"]) else (", ".join(failed) if failed else "unknown")

    final_formula = {
        "raw":      "spike_raw & !anti & step_ok & sustain_ok & stop_ok",
        "est":      "spike_est & !anti & step_ok & sustain_ok & stop_ok",
        "both_or":  "spike_any(OR) & !anti & step_ok & sustain_ok & stop_ok",
        "both_and": "spike_both(AND) & !anti & step_ok & sustain_ok & stop_ok",
    }[s["cand_mode"]]

    rows.append({
        "Check": "FINAL DECISION",
        "Ideal/Threshold": final_formula,
        "Actual": str(bool(s["decision"])),
        "Pass": bool(s["decision"]),
        "Details": reason
    })

    tbl = pd.DataFrame(rows, columns=["Check", "Ideal/Threshold", "Actual", "Pass", "Details"])

    if pretty_print:
        for c in ["Ideal/Threshold", "Actual"]:
            tbl[c] = tbl[c].apply(_fmt_num)
        tbl["Pass"] = tbl["Pass"].map(lambda v: "✅" if bool(v) else "❌")
        print("\n" + "="*80)
        print(f"Refuel decision @ {pd.to_datetime(s['when'])}")
        print("="*80)
        print(tbl.to_string(index=False))
        print("="*80 + "\n")

    return tbl

# ----------------------- PF-based refuel event detection -----------------------
def build_event_track(df: pd.DataFrame, events_pf: pd.DataFrame,
                      t_col="datetime", out_col="pf_event_bin") -> pd.DataFrame:
    """
    Create a 0/1 event track aligned to df[t_col] from the events table.
    """
    w = df.copy()
    mask = np.zeros(len(w), dtype=int)
    if events_pf is not None and not events_pf.empty:
        t = pd.to_datetime(w[t_col]).to_numpy()
        for _, e in events_pf.iterrows():
            m = (t >= np.datetime64(e["start"])) & (t <= np.datetime64(e["end"]))
            mask[m] = 1
    w[out_col] = mask
    return w

def detect_refuel_events_pf_robust(
    w: pd.DataFrame,
    t_col="datetime",
    p_col="pf_p_refuel",
    pf_fuel_col="fuel_est_pf",    # <-- your column name
    raw_fuel_col="fuel",
    rate_col="fuel_rate_pf",
    p_threshold=0.65,             # prob gate
    max_hole_s=120,               # allow short dips inside an event
    min_duration_s=60,
    min_volume_l=2.0,             # rate integral
    min_pf_step_l=0.8,            # PF step fallback
    min_raw_step_l=0.8,           # RAW step fallback
    prob_area_min=25.0,           # ∑ p·dt fallback
    merge_gap_s=1800,             # merge < 30 min apart
    pre_win_s=120, post_win_s=120 # RAW step medians
) -> pd.DataFrame:
    df = w[[t_col, p_col, pf_fuel_col, raw_fuel_col, rate_col]].copy()
    df[t_col] = pd.to_datetime(df[t_col])
    df = df.sort_values(t_col).reset_index(drop=True)

    t = df[t_col].to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0).to_numpy()
    pf = pd.to_numeric(df[pf_fuel_col], errors="coerce").to_numpy()
    raw = pd.to_numeric(df[raw_fuel_col], errors="coerce").to_numpy()
    rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).to_numpy()

    if len(t) < 2:
        return pd.DataFrame(columns=['start','end','mid','duration_s','vol_l','mean_prob','max_prob'])

    # time deltas (seconds)
    dt = np.diff(t).astype('timedelta64[s]').astype(int)
    dt = np.r_[dt[0], dt]

    # 1) initial mask from probability
    mask = p >= p_threshold

    # 2) bridge short dips (morphological closing in time)
    #    if a false run (gap) is shorter than max_hole_s, fill it.
    i = 0
    while i < len(mask):
        if not mask[i]:
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            # gap is [i, j-1]
            gap_seconds = int(np.sum(dt[i:j])) if j > i else 0
            left_true  = (i-1 >= 0) and mask[i-1]
            right_true = (j < len(mask)) and mask[j]
            if left_true and right_true and gap_seconds <= max_hole_s:
                mask[i:j] = True
            i = j
        else:
            i += 1

    # 3) extract contiguous true runs as candidates
    events = []
    in_evt = False
    s = 0
    for k, val in enumerate(mask):
        if val and not in_evt:
            in_evt = True
            s = k
        elif (not val) and in_evt:
            e = k - 1
            start_t, end_t = t[s], t[e]
            dur = int((end_t - start_t).astype('timedelta64[s]').astype(int))
            sl = slice(s, e+1)

            # metrics
            mean_prob = float(np.mean(p[sl]))
            max_prob  = float(np.max(p[sl]))
            prob_area = float(np.sum(p[sl] * dt[sl]))

            vol_int = float(np.sum(np.maximum(rate[sl], 0.0) * dt[sl]))  # L (rate is L/s)
            pf_step = float(max(0.0, pf[e] - pf[s]))

            # RAW step via medians
            def median_in_window(t0, width_s, side):
                if side == "pre":
                    m = (t >= t0 - np.timedelta64(width_s, 's')) & (t < t0)
                else:
                    m = (t > t0) & (t <= t0 + np.timedelta64(width_s, 's'))
                v = raw[m]
                v = v[np.isfinite(v)]
                return float(np.median(v)) if v.size >= 3 else np.nan

            pre_med  = median_in_window(start_t, pre_win_s, "pre")
            post_med = median_in_window(end_t,   post_win_s, "post")
            raw_step = float(max(0.0, post_med - pre_med)) if (np.isfinite(pre_med) and np.isfinite(post_med)) else 0.0

            # acceptance: ANY of the signals is strong enough
            accept = (dur >= min_duration_s) and (
                (vol_int >= min_volume_l) or
                (pf_step >= min_pf_step_l) or
                (raw_step >= min_raw_step_l) or
                (prob_area >= prob_area_min)
            )

            if accept:
                events.append({
                    'start': pd.Timestamp(start_t),
                    'end':   pd.Timestamp(end_t),
                    'mid':   pd.Timestamp(start_t + (end_t - start_t)/2),
                    'duration_s': dur,
                    'vol_l': max(vol_int, pf_step, raw_step),
                    'mean_prob': mean_prob,
                    'max_prob': max_prob,
                    'pf_step_l': pf_step,
                    'raw_step_l': raw_step,
                    'prob_area': prob_area
                })
            in_evt = False

    if in_evt:
        e = len(mask) - 1
        start_t, end_t = t[s], t[e]
        dur = int((end_t - start_t).astype('timedelta64[s]').astype(int))
        sl = slice(s, e+1)
        mean_prob = float(np.mean(p[sl])); max_prob = float(np.max(p[sl]))
        prob_area = float(np.sum(p[sl] * dt[sl]))
        vol_int = float(np.sum(np.maximum(rate[sl], 0.0) * dt[sl]))
        pf_step = float(max(0.0, pf[e] - pf[s]))
        # simple raw step
        def med(t0, width_s, side):
            if side == "pre":
                m = (t >= t0 - np.timedelta64(width_s, 's')) & (t < t0)
            else:
                m = (t > t0) & (t <= t0 + np.timedelta64(width_s, 's'))
            v = raw[m]; v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size >= 3 else np.nan
        pre_med, post_med = med(start_t, pre_win_s, "pre"), med(end_t, post_win_s, "post")
        raw_step = float(max(0.0, post_med - pre_med)) if (np.isfinite(pre_med) and np.isfinite(post_med)) else 0.0

        accept = (dur >= min_duration_s) and (
            (vol_int >= min_volume_l) or (pf_step >= min_pf_step_l) or
            (raw_step >= min_raw_step_l) or (prob_area >= prob_area_min)
        )
        if accept:
            events.append({
                'start': pd.Timestamp(start_t),
                'end':   pd.Timestamp(end_t),
                'mid':   pd.Timestamp(start_t + (end_t - start_t)/2),
                'duration_s': dur,
                'vol_l': max(vol_int, pf_step, raw_step),
                'mean_prob': mean_prob,
                'max_prob': max_prob,
                'pf_step_l': pf_step,
                'raw_step_l': raw_step,
                'prob_area': prob_area
            })

    if not events:
        return pd.DataFrame(columns=['start','end','mid','duration_s','vol_l','mean_prob','max_prob'])

    ev = pd.DataFrame(events).sort_values('start').reset_index(drop=True)

    # merge nearby
    merged = []
    cur = ev.iloc[0].to_dict()
    for i in range(1, len(ev)):
        gap = (ev.loc[i, 'start'] - cur['end']).total_seconds()
        if gap <= merge_gap_s:
            cur['end'] = max(cur['end'], ev.loc[i, 'end'])
            cur['mid'] = cur['start'] + (cur['end'] - cur['start'])/2
            cur['duration_s'] = (cur['end'] - cur['start']).total_seconds()
            cur['vol_l'] = cur['vol_l'] + ev.loc[i, 'vol_l']
            cur['mean_prob'] = 0.5*(cur['mean_prob'] + ev.loc[i, 'mean_prob'])
            cur['max_prob']  = max(cur['max_prob'], ev.loc[i, 'max_prob'])
        else:
            merged.append(cur)
            cur = ev.iloc[i].to_dict()
    merged.append(cur)
    ev = pd.DataFrame(merged)

    return ev[['start','end','mid','duration_s','vol_l','mean_prob','max_prob']]


def explain_window(
    w: pd.DataFrame,
    t0, t1,
    t_col="datetime",
    p_col="pf_p_refuel",
    rate_col="fuel_rate_pf",
    pf_fuel_col="fuel_est_pf",
    raw_fuel_col="fuel",
    # use the SAME thresholds you use in your detector:
    min_duration_s=60,
    min_volume_l=2.0,
    min_pf_step_l=0.8,
    min_raw_step_l=0.8,
    prob_area_min=25.0,
):
    import numpy as np
    import pandas as pd

    t = pd.to_datetime(w[t_col])
    t0 = pd.Timestamp(t0); t1 = pd.Timestamp(t1)
    m = (t >= t0) & (t <= t1)
    ww = w.loc[m].copy()
    if ww.empty:
        print("No rows in that window."); 
        return

    # vectors
    tt   = pd.to_datetime(ww[t_col]).to_numpy()
    p    = pd.to_numeric(ww[p_col], errors="coerce").fillna(0.0).to_numpy()
    rate = pd.to_numeric(ww[rate_col], errors="coerce").fillna(0.0).to_numpy()
    pf   = pd.to_numeric(ww[pf_fuel_col], errors="coerce").to_numpy()
    raw  = pd.to_numeric(ww[raw_fuel_col], errors="coerce").to_numpy()

    # dt in seconds (same length as series)
    dt = np.diff(tt).astype("timedelta64[s]").astype(int)
    if dt.size == 0:
        dt = np.array([0])
    dt = np.r_[dt[0], dt]

    duration_s  = int((tt[-1] - tt[0]).astype("timedelta64[s]").astype(int))
    prob_area   = float(np.sum(p * dt))
    vol_int_l   = float(np.sum(np.maximum(rate, 0.0) * dt))
    pf_step_l   = float(max(0.0, (pf[-1] - pf[0])))

    # raw step via medians (robust)
    pre_med  = float(np.median(raw[:max(3, len(raw)//5)])) if len(raw) >= 3 else np.nan
    post_med = float(np.median(raw[-max(3, len(raw)//5):])) if len(raw) >= 3 else np.nan
    raw_step_l = float(max(0.0, post_med - pre_med)) if np.isfinite(pre_med) and np.isfinite(post_med) else np.nan

    mean_p = float(np.mean(p)); max_p = float(np.max(p))

    # which tests pass?
    tests = {
        "dur_ok": duration_s >= min_duration_s,
        "vol_ok": vol_int_l >= min_volume_l,
        "pf_step_ok": pf_step_l >= min_pf_step_l,
        "raw_step_ok": (raw_step_l >= min_raw_step_l) if np.isfinite(raw_step_l) else False,
        "area_ok": prob_area >= prob_area_min,
    }
    accept = tests["dur_ok"] and (tests["vol_ok"] or tests["pf_step_ok"] or tests["raw_step_ok"] or tests["area_ok"])

    print("---- explain_window ----")
    print(f"t0={tt[0]}  t1={tt[-1]}  duration_s={duration_s}")
    print(f"mean_p={mean_p:.3f}  max_p={max_p:.3f}  prob_area={prob_area:.1f}")
    print(f"vol_int_l={vol_int_l:.2f}  pf_step_l={pf_step_l:.2f}  raw_step_l={raw_step_l:.2f}")
    print("tests:", tests)
    print("ACCEPTED?" , accept)

def explain_around(w, center_ts, minutes=10, **kw):
    c = pd.Timestamp(center_ts)
    return explain_window(w, c - pd.Timedelta(minutes=minutes), c + pd.Timedelta(minutes=minutes), **kw)



def detect_refuel_events_prob_area(
    w: pd.DataFrame,
    t_col="datetime",
    p_col="pf_p_refuel",        # [0..1]
    rate_col="fuel_rate_pf",    # L/s  (not L/h)
    fuel_col="pf_fuel",         # PF-smoothed fuel
    raw_fuel_col="fuel",        # optional: raw fuel
    p_threshold=0.70,
    min_duration_s=30,
    area_min=25.0,              # Σ p*dt  (sec)
    min_pf_step_l=1.0,          # PF step across window
    min_volume_l=2.0,           # ∫ max(rate,0) dt
    min_raw_step_l=0.0,         # set to ~0.8 if you want raw confirmation
    bridge_gap_s=180,           # <-- join spikes ≤ 3 min apart
    pre_pad_s=60,               # <-- include 1 min before
    post_pad_s=60,              # <-- include 1 min after
    merge_gap_s=1800,           # merge accepted events < 30 min apart
):
    df = w[[t_col, p_col]].copy()
    df[t_col] = pd.to_datetime(df[t_col])
    df = df.sort_values(t_col).reset_index(drop=True)

    t = df[t_col].to_numpy()
    if len(t) < 2:
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    p = pd.to_numeric(w[p_col], errors="coerce").reindex(df.index).fillna(0.0).to_numpy()
    rate = pd.to_numeric(w.get(rate_col, pd.Series(0, index=w.index)), errors="coerce").reindex(df.index).fillna(0.0).to_numpy()
    fuel = pd.to_numeric(w.get(fuel_col, pd.Series(np.nan, index=w.index)), errors="coerce").reindex(df.index).to_numpy()
    raw  = pd.to_numeric(w.get(raw_fuel_col, pd.Series(np.nan, index=w.index)), errors="coerce").reindex(df.index).to_numpy()

    # cadence + helpers
    dt_all = np.diff(t).astype("timedelta64[s]").astype(int)
    cadence_s = int(np.median(dt_all)) if dt_all.size else 60

    def idx_from_time(base_idx, seconds, direction):
        off = int(round(seconds / cadence_s))
        if direction == "pre":
            return max(0, base_idx - off)
        else:
            return min(len(t)-1, base_idx + off)

    # 1) initial “high p” runs
    high = p >= p_threshold
    if not np.any(high):
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    idx = np.flatnonzero(high)
    brk = np.where(np.diff(idx) > 1)[0] + 1
    se = [(int(b[0]), int(b[-1])) for b in np.split(idx, brk) if b.size]

    # 2) bridge close runs BEFORE acceptance
    bridged = []
    if se:
        cur_s, cur_e = se[0]
        for s, e in se[1:]:
            gap_s = int((t[s] - t[cur_e]).astype("timedelta64[s]").astype(int))
            if gap_s <= bridge_gap_s:
                cur_e = e  # join
            else:
                bridged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        bridged.append((cur_s, cur_e))
    else:
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    rows = []
    for s, e in bridged:
        # 3) pad window around the run for step/area/volume
        s_pad = idx_from_time(s, pre_pad_s, "pre")
        e_pad = idx_from_time(e, post_pad_s, "post")

        duration_s = int((t[e_pad] - t[s_pad]).astype("timedelta64[s]").astype(int))
        if duration_s < min_duration_s:
            continue

        # per-sample dt (same length as slice)
        dt_run = np.diff(t[s_pad:e_pad+1]).astype("timedelta64[s]").astype(int)
        if dt_run.size == 0:
            dt_run = np.array([cadence_s], int)
        else:
            dt_run = np.r_[dt_run[0], dt_run]

        # 4) metrics
        prob_area = float(np.sum(p[s_pad:e_pad+1] * dt_run))
        vol_int   = float(np.sum(np.maximum(rate[s_pad:e_pad+1], 0.0) * dt_run))

        pf_step = 0.0
        if np.isfinite(fuel[s_pad]) and np.isfinite(fuel[e_pad]):
            pf_step = float(max(0.0, fuel[e_pad] - fuel[s_pad]))

        raw_step = 0.0
        if np.isfinite(raw[s_pad]) and np.isfinite(raw[e_pad]):
            raw_step = float(max(0.0, raw[e_pad] - raw[s_pad]))

        accept = (
            (prob_area >= area_min) or
            (pf_step   >= min_pf_step_l) or
            (vol_int   >= min_volume_l) or
            (raw_step  >= min_raw_step_l)
        )
        if not accept:
            continue

        rows.append({
            "start": pd.Timestamp(t[s_pad]),
            "end":   pd.Timestamp(t[e_pad]),
            "mid":   pd.Timestamp(t[s_pad] + (t[e_pad] - t[s_pad]) / 2),
            "duration_s": duration_s,
            "vol_l": max(vol_int, pf_step, raw_step),
            "mean_prob": float(np.mean(p[s_pad:e_pad+1])),
            "max_prob":  float(np.max(p[s_pad:e_pad+1])),
            "prob_area": prob_area
        })

    if not rows:
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    ev = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

    # 5) merge accepted events if still close
    if len(ev) > 1:
        merged = []
        cur = ev.iloc[0].to_dict()
        for i in range(1, len(ev)):
            gap = (ev.loc[i, "start"] - cur["end"]).total_seconds()
            if gap <= merge_gap_s:
                cur["end"] = max(cur["end"], ev.loc[i, "end"])
                cur["mid"] = cur["start"] + (cur["end"] - cur["start"]) / 2
                cur["duration_s"] = int((cur["end"] - cur["start"]).total_seconds())
                cur["vol_l"] += ev.loc[i, "vol_l"]
                cur["mean_prob"] = (cur["mean_prob"] + ev.loc[i, "mean_prob"]) / 2.0
                cur["max_prob"]  = max(cur["max_prob"], ev.loc[i, "max_prob"])
                cur["prob_area"] += ev.loc[i, "prob_area"]
            else:
                merged.append(cur); cur = ev.iloc[i].to_dict()
        merged.append(cur)
        ev = pd.DataFrame(merged)

    return ev[["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"]]