import argparse
from .io import read_standardize, make_work
from .estimates import add_estimates
from .events import build_event_track, detect_refuel_events_prob_area
from .visualisation import plot_detailed_estimates
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Fuel estimation comparison (no event detection)")
    p.add_argument("--csv", required=True, help="Path to fuel data CSV")
    p.add_argument("--save-html", default=None, help="Optional: save plot to HTML file")
    
    args = p.parse_args()
    
    # Load and prepare data
    print(f"Reading data from {args.csv}...")
    df = read_standardize(args.csv)
    work = make_work(df)
    
    # Add all estimates (RM, Kalman 1D/2D, PF) with tuned params
    print("Computing estimates...")
    work = add_estimates(work)

    events_pf = detect_refuel_events_prob_area(
        work,
        p_threshold=0.75,              
        rate_threshold=0.0018,         # ~6.5 L/h
        step_window_s=180,
        bridge_gap_s=180,              # join spikes ≤5 min
        pre_pad_s=60, post_pad_s=60,
        area_min=20.0, min_pf_step_l=1.0, min_volume_l=2.0, min_raw_step_l=0.0
    )
    work = build_event_track(work, events_pf, out_col="pf_event_bin")
    fig = plot_detailed_estimates(work, events_pf=events_pf, title="Fuel Estimation Methods - Detailed View", height=1000)

    if not events_pf.empty:
        print("\nPF Refuel Events:")
        # print(events_pf[["start","end","duration_s","vol_l","mean_prob","max_prob"]].to_string(index=False))
    
    # Save or show
    if args.save_html:
        print(f"Saving plot to {args.save_html}...")
        fig.write_html(args.save_html)
        print("Done!")
    
    print("Displaying interactive plot...")
    fig.show()

if __name__ == "__main__":
    main()