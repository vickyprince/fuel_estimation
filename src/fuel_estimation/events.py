import numpy as np
import pandas as pd
from typing import Optional

from typing import Optional, Tuple, Set

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

import numpy as np, pandas as pd

def detect_refuel_events_prob_area(
    w: pd.DataFrame,
    t_col="datetime", p_col="pf_p_refuel",
    rate_col="fuel_rate_pf", fuel_col="pf_fuel", raw_fuel_col="fuel",
    p_threshold=0.70,
    rate_threshold=0.0018,      # ≈ 6.5 L/h  (PF rate is L/s)
    step_window_s=180,          # look ±3 min for PF step
    min_duration_s=30,
    area_min=25.0,              # Σ p*dt  (sec)
    min_pf_step_l=1.0,
    min_volume_l=2.0,
    min_raw_step_l=0.0,
    bridge_gap_s=180,           # join spikes ≤ 3 min apart
    pre_pad_s=60, post_pad_s=60,
    merge_gap_s=1800
):
    """
    Detect refuel events based on PF refuel probability area, rate, and fuel steps.
    if any of: big probability area, clear step in PF/RAW, or enough integrated positive rate, plus minimal duration.
    Returns a DataFrame with detected events and their properties.
    """
    df = w[[t_col, p_col]].copy()
    df[t_col] = pd.to_datetime(df[t_col])
    df = df.sort_values(t_col).reset_index(drop=True)

    t = df[t_col].to_numpy()
    if len(t) < 2:
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    p    = pd.to_numeric(w[p_col], errors="coerce").reindex(df.index).fillna(0.0).to_numpy()
    rate = pd.to_numeric(w.get(rate_col, pd.Series(0, index=w.index)), errors="coerce").reindex(df.index).fillna(0.0).to_numpy()
    fuel = pd.to_numeric(w.get(fuel_col, pd.Series(np.nan, index=w.index)), errors="coerce").reindex(df.index).to_numpy()
    raw  = pd.to_numeric(w.get(raw_fuel_col, pd.Series(np.nan, index=w.index)), errors="coerce").reindex(df.index).to_numpy()

    dt_all = np.diff(t).astype("timedelta64[s]").astype(int)
    cadence_s = int(np.median(dt_all)) if dt_all.size else 60

    # --- seed paths ---
    prob_seed = p >= p_threshold

    rate_med = pd.Series(rate).rolling(3, center=True, min_periods=1).median().to_numpy()
    rate_seed = rate_med > rate_threshold

    W = max(1, int(round(step_window_s / max(1, cadence_s))))
    f = pd.Series(fuel).interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).median().to_numpy()
    pre  = np.r_[np.full(W, f[0]),  f[:-W]]
    post = np.r_[f[W:],            np.full(W, f[-1])]
    step_seed = (post - pre) >= min_pf_step_l

    seed = prob_seed | rate_seed | step_seed
    if not np.any(seed):
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    idx = np.flatnonzero(seed)
    brk = np.where(np.diff(idx) > 1)[0] + 1
    runs = [ (int(b[0]), int(b[-1])) for b in np.split(idx, brk) if b.size ]

    # bridge close runs
    bridged = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        gap_s = int((t[s] - t[cur_e]).astype("timedelta64[s]").astype(int))
        if gap_s <= bridge_gap_s:
            cur_e = e
        else:
            bridged.append((cur_s, cur_e)); cur_s, cur_e = s, e
    bridged.append((cur_s, cur_e))

    def shift_idx(i, secs, direction):
        off = int(round(secs / max(1, cadence_s)))
        return max(0, i-off) if direction=="pre" else min(len(t)-1, i+off)

    rows = []
    for s, e in bridged:
        s0 = shift_idx(s, pre_pad_s, "pre")
        e0 = shift_idx(e, post_pad_s, "post")
        dur = int((t[e0] - t[s0]).astype("timedelta64[s]").astype(int))
        if dur < min_duration_s: 
            continue

        dtr = np.diff(t[s0:e0+1]).astype("timedelta64[s]").astype(int)
        dtr = np.r_[dtr[0] if dtr.size else cadence_s, dtr]  # same length as slice

        prob_area = float(np.sum(p[s0:e0+1] * dtr))
        vol_int   = float(np.sum(np.maximum(rate[s0:e0+1], 0.0) * dtr))
        pf_step   = float(max(0.0, fuel[e0] - fuel[s0])) if np.isfinite(fuel[s0]) and np.isfinite(fuel[e0]) else 0.0
        raw_step  = float(max(0.0, raw[e0]  - raw[s0]))  if np.isfinite(raw[s0])  and np.isfinite(raw[e0])  else 0.0

        accept = (prob_area >= area_min) or (pf_step >= min_pf_step_l) or (vol_int >= min_volume_l) or (raw_step >= min_raw_step_l)
        if not accept: 
            continue

        rows.append({
            "start": pd.Timestamp(t[s0]), "end": pd.Timestamp(t[e0]),
            "mid": pd.Timestamp(t[s0] + (t[e0] - t[s0])/2),
            "duration_s": dur,
            "vol_l": max(vol_int, pf_step, raw_step),
            "mean_prob": float(np.mean(p[s0:e0+1])),
            "max_prob": float(np.max(p[s0:e0+1])),
            "prob_area": prob_area
        })

    if not rows:
        return pd.DataFrame(columns=["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"])

    ev = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    # merge nearby accepted events
    if len(ev) > 1:
        out = []
        cur = ev.iloc[0].to_dict()
        for i in range(1, len(ev)):
            gap = (ev.loc[i,"start"] - cur["end"]).total_seconds()
            if gap <= merge_gap_s:
                cur["end"] = max(cur["end"], ev.loc[i,"end"])
                cur["mid"] = cur["start"] + (cur["end"] - cur["start"])/2
                cur["duration_s"] = int((cur["end"] - cur["start"]).total_seconds())
                cur["vol_l"] += ev.loc[i,"vol_l"]
                cur["mean_prob"] = (cur["mean_prob"] + ev.loc[i,"mean_prob"]) / 2.0
                cur["max_prob"] = max(cur["max_prob"], ev.loc[i,"max_prob"])
                cur["prob_area"] += ev.loc[i,"prob_area"]
            else:
                out.append(cur); cur = ev.iloc[i].to_dict()
        out.append(cur); ev = pd.DataFrame(out)

    return ev[["start","end","mid","duration_s","vol_l","mean_prob","max_prob","prob_area"]]