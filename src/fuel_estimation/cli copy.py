import argparse
from .io import read_standardize, make_work
from .estimates import add_estimates
from .events import detect_refuel_events, explain_refuel_table
from .validate import validate_refuels
from .visualisation import timeline_with_events, eda_dashboard, align_pass_fail, timeline_all_estimates
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Fuel refuel detector (no files written unless --save)")
    p.add_argument("--csv", required=True)
    p.add_argument("--show-eda", action="store_true")
    p.add_argument("--save-html", default=None, help="Optional: save timeline to this HTML")
    p.add_argument("--pre", default="7min"); p.add_argument("--post", default="7min")
    p.add_argument("--z", type=float, default=3.0); p.add_argument("--min-frac", type=float, default=0.03)
    args = p.parse_args()

    df = read_standardize(args.csv)
    work = make_work(df)

    # add all estimates (RM, Kalman 1D/2D, PF) with tuned params
    work = add_estimates(work)

    # rule-based detector params (keep your logic; tweak if you wish)
    DETECT_PARAMS = dict(
        q=0.98, window_pts=11, pre="10min", post="10min",
        z=2.7, min_frac=0.03, cand_source="both_and",
        anti_glitch_window="3min", anti_glitch_ratio=0.7,
        speed_col="cmd_speed", stop_eps=0.05, stop_frac=0.80,
        stop_pre="6min", stop_post="3min",
        require_stop=True
    )
    events = detect_refuel_events(work, **DETECT_PARAMS)

    fig_rule_based = timeline_with_events(work, events, title="Fuel + events (validated)")
    # quick look (like before, but with PF line included):
    fig = timeline_with_events(work, events, title="Fuel + events (all estimates)")
    fig.show()

    # deeper understanding (fuel + rates + PF prob):
    fig3 = timeline_all_estimates(work, events, title="Fuel, Rates and PF Probability (best params)")
    fig3.show()
    if args.save_html:
        fig_rule_based.write_html(args.save_html.replace(".html","_rule_based.html"), auto_open=True)
    else:
        pio.renderers.default = "browser"
        fig_rule_based.show()

    # if args.show_eda:
    #     dash_fig = eda_dashboard(df, work, events)
    #     dash_fig.show()
    #     if args.save_html:
    #         dash_fig.write_html(args.save_html.replace(".html","_eda.html"), auto_open=True)

    # import plotly.graph_objects as go

    # # Add HMM probabilities
    # from .hmm import add_hmm_to_work_strict, hmm_events_from_probs_ultra_strict

    # work = add_hmm_to_work_strict(work, speed_col="cmd_speed", stop_eps=0.05)

    # events = hmm_events_from_probs_ultra_strict(
    #     work,
    #     prob_col="hmm_p_refuel",
    #     thr_on=0.98,          # Very high threshold
    #     thr_off=0.85,         # Strong hysteresis
    #     min_duration="8min",  # Sustained increase required
    #     merge_gap="20min",
    #     estimate_col="fuel_est_kalman",
    #     speed_col="cmd_speed",
    #     stop_eps=0.05,
    #     min_step=50.0        # Minimum 50L increase
    # )


    # print(f"Found {len(events)} refuel events")
    # print(events)

    # # Quick plot
    # fig_hmm = go.Figure()
    # t = pd.to_datetime(work["datetime"])
    # fuel = pd.to_numeric(work["fuel"], errors="coerce")
    # p_refuel = pd.to_numeric(work["hmm_p_refuel"], errors="coerce")

    # fig_hmm.add_trace(go.Scatter(x=t, y=fuel, name="Fuel", mode="lines"))
    # fig_hmm.add_trace(go.Scatter(
    #     x=t, y=p_refuel, name="P(refuel)", mode="lines",
    #     yaxis="y2", fill="tozeroy", opacity=0.3
    # ))

    # if not events.empty:
    #     evt_fuel = np.interp(
    #         pd.to_datetime(events["datetime"]).astype(int),
    #         t.astype(int),
    #         fuel.values
    #     )
    #     fig_hmm.add_trace(go.Scatter(
    #         x=events["datetime"], y=evt_fuel,
    #         mode="markers", marker=dict(color="green", size=10),
    #         name="Refuels"
    #     ))

    # fig_hmm.update_layout(
    #     title="Fuel + HMM Refuel Events",
    #     template="plotly_white",
    #     yaxis=dict(title="Fuel"),
    #     yaxis2=dict(title="P(refuel)", overlaying="y", side="right", range=[0, 1]),
    #     hovermode="x unified"
    # )
    # fig_hmm.show()
    # if args.save_html:
    #     fig_hmm.write_html(args.save_html.replace(".html","_hmm.html"), auto_open=True)

if __name__ == "__main__":
    main()