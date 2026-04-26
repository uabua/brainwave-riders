from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import TARGET_FREQS

FREQ_RANGE = (5, 50)
LINE_COLOR = "#1f77b4"


def psd_per_class(
    psd_df: pd.DataFrame,
    subject: int,
    *,
    target_freqs: list[int] = TARGET_FREQS,
) -> go.Figure:
    sub_df = psd_df[psd_df["subject"] == subject]
    sessions = sorted(sub_df["session"].unique())
    n_rows = len(target_freqs)
    n_cols = len(sessions)

    titles = []
    for r in range(n_rows):
        for c, sess in enumerate(sessions):
            titles.append(f"Session {sess}" if r == 0 else "")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes="rows",
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
        subplot_titles=titles,
    )

    for r, target in enumerate(target_freqs, start=1):
        for c, sess in enumerate(sessions, start=1):
            cell = sub_df[
                (sub_df["session"] == sess) & (sub_df["target_hz"] == target)
            ].sort_values("freq")
            fig.add_trace(
                go.Scatter(
                    x=cell["freq"],
                    y=cell["psd"],
                    mode="lines",
                    line=dict(color=LINE_COLOR, width=1.5),
                    showlegend=False,
                    hovertemplate=(
                        f"Sess {sess} • target {target} Hz<br>"
                        "%{x:.2f} Hz<br>%{y:.3g}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
            for sf in target_freqs:
                is_target = sf == target
                fig.add_vline(
                    x=sf,
                    line=dict(
                        color="red" if is_target else "gray",
                        dash="dash",
                        width=2 if is_target else 0.8,
                    ),
                    opacity=0.7,
                    row=r,
                    col=c,
                )
            if c == 1:
                fig.update_yaxes(
                    title_text=f"{int(target)} Hz",
                    row=r,
                    col=c,
                )

    fig.update_xaxes(range=list(FREQ_RANGE))
    for c in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Frequency (Hz)", row=n_rows, col=c)

    fig.update_layout(
        title=f"Subject {subject} — PSD per class (channel-averaged)",
        height=180 * n_rows + 80,
        margin=dict(l=70, r=20, t=80, b=50),
    )
    return fig
