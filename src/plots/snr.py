from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import CH_NAMES, TARGET_FREQS


def snr_matrix(
    snr_df: pd.DataFrame,
    subject: int,
    *,
    target_freqs: list[int] = TARGET_FREQS,
    ch_names: list[str] = CH_NAMES,
) -> go.Figure:
    """Heatmap per session: rows=target frequency, cols=channel, value=mean SNR (h=1)."""
    sub = snr_df[snr_df["subject"] == subject]
    sessions = sorted(sub["session"].unique())

    fig = make_subplots(
        rows=1,
        cols=len(sessions),
        shared_yaxes=True,
        subplot_titles=[f"Session {s}" for s in sessions],
        horizontal_spacing=0.08,
    )

    pivots = []
    for sess in sessions:
        ss = sub[sub["session"] == sess]
        pivot = ss.pivot(index="target_hz", columns="channel", values="snr")
        pivot = pivot.reindex(index=target_freqs, columns=ch_names)
        pivots.append(pivot)

    zmax = max(p.to_numpy().max() for p in pivots if p.notna().any().any())

    for c, (sess, pivot) in enumerate(zip(sessions, pivots), start=1):
        fig.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=pivot.columns.tolist(),
                y=[f"{int(t)} Hz" for t in pivot.index],
                colorscale="Viridis",
                zmin=0,
                zmax=float(zmax),
                colorbar=dict(title="SNR") if c == len(sessions) else None,
                showscale=(c == len(sessions)),
                hovertemplate=(
                    f"Sess {sess}<br>%{{x}} • target %{{y}}<br>"
                    "SNR = %{z:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=c,
        )
        fig.update_xaxes(title_text="Channel", row=1, col=c)
    fig.update_yaxes(title_text="Target", row=1, col=1, autorange="reversed")

    fig.update_layout(
        title=f"Subject {subject} — SNR matrix (target × channel, h=1)",
        height=320,
        margin=dict(l=70, r=20, t=70, b=50),
    )
    return fig
