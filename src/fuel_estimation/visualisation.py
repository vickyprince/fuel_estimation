import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_detailed_estimates(work: pd.DataFrame, events_pf: pd.DataFrame = None,
                            title: str = "Fuel Estimation Methods - Detailed View",
                            height: int = 1500):
    """
    3 equal-height rows:
      1) Fuel levels (raw & estimates) + shaded refuel bands
      2) Rates (L/h), y-limited to ±10 L/h
      3) PF refueling (probability & 0/1 events) + event volume markers
    """
    w = work.copy()
    w["datetime"] = pd.to_datetime(w["datetime"])
    w = w.sort_values("datetime").reset_index(drop=True)

    t = w["datetime"]
    y = w["fuel"].astype(float)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Equal heights for all three rows
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[1/3, 1/3, 1/3],
        subplot_titles=("Fuel Levels (raw & estimates)",
                        "Consumption Rates (L/h)",
                        "PF Refueling (probability & events)")
    )

    # Row 1: Fuel levels
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Fuel (raw)",
                             line=dict(color="lightgray", width=1)), row=1, col=1)
    if "fuel_est_kalman" in w:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_est_kalman"], mode="lines",
                                 name="Kalman 1D", line=dict(dash="dash")), row=1, col=1)
    if "fuel_est_kalman_2D_smooth" in w:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_est_kalman_2D_smooth"], mode="lines",
                                 name="Kalman 2D (smoothed)", line=dict(width=2.5)), row=1, col=1)
    if "fuel_est_pf" in w:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_est_pf"], mode="lines",
                                 name="Particle Filter", line=dict(width=2.5, color="orange")), row=1, col=1)

    # Shade PF refuel events on row 1
    if events_pf is not None and not events_pf.empty:
        for _, e in events_pf.iterrows():
            fig.add_vrect(x0=e["start"], x1=e["end"], row=1, col=1,
                          fillcolor="tomato", opacity=0.15, line_width=0)

    # Row 2: Rates in L/h, limited to ±10 L/h
    if "fuel_rate_kalman_2D_smooth_lph" in w:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_rate_kalman_2D_smooth_lph"], mode="lines",
                                 name="Rate (Kalman 2D smooth, L/h)", line=dict(color="purple")), row=2, col=1)
    if "fuel_rate_pf_lph" in w:
        fig.add_trace(go.Scatter(x=t, y=w["fuel_rate_pf_lph"], mode="lines",
                                 name="Rate (PF, L/h)", line=dict(color="orange")), row=2, col=1)
    fig.update_yaxes(title_text="Rate (L/h)", row=2, col=1, range=[-10, 10], zeroline=True)

    # fig.update_yaxes(
    #     title_text="Rate (L/h)",
    #     row=2, col=1,
    #     autorange=True,            # let Plotly set min/max from the data
    #     tickmode="linear",         # use a constant tick step
    #     dtick=10,                  # one major tick every 10 L/h
    #     zeroline=True              # keep the zero line
    # )

    # Row 3: PF probability + 0/1 event curve + event markers
    if "pf_p_refuel" in w:
        fig.add_trace(go.Scatter(x=t, y=w["pf_p_refuel"], mode="lines",
                                 name="PF refuel probability", line=dict(color="teal")), row=3, col=1)
    if "pf_event_bin" in w:
        fig.add_trace(go.Scatter(
            x=t, y=w["pf_event_bin"], name="PF event (0/1)",
            mode="lines", line=dict(color="firebrick", width=2), fill="tozeroy", opacity=0.25
        ), row=3, col=1)
    if events_pf is not None and not events_pf.empty:
        fig.add_trace(go.Scatter(
            x=events_pf["mid"], y=[1.0]*len(events_pf),
            mode="markers+text",
            text=[f"+{v:.1f} L" for v in events_pf["vol_l"]],
            textposition="top center",
            name="PF event volume",
            marker=dict(color="firebrick", size=8)
        ), row=3, col=1)

    # Axes labels
    fig.update_yaxes(title_text="Fuel (L)", row=1, col=1)
    fig.update_yaxes(title_text="Refuel (prob / event)", row=3, col=1, range=[0, 1.05], dtick=0.5)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    # Legend: vertical, outside on the right
    fig.update_layout(
        template="plotly_white",
        title=title,
        hovermode="x unified",
        height=height,
        legend=dict(
            orientation="v",
            x=1.02, y=1.0,           # outside top-right
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=70, r=200),  # r gives room for outside legend
        xaxis3=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig