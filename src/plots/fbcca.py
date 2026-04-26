from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import TARGET_FREQS


def fbcca_matrix(
    fbcca_df: pd.DataFrame,
    subject: int,
    *,
    target_freqs: list[int] = TARGET_FREQS,
) -> go.Figure:
    """Trial × reference-frequency FBCCA score heatmap, trials sorted by true target."""
    sub = fbcca_df[fbcca_df["subject"] == subject]
    sessions = sorted(sub["session"].unique())

    fig = make_subplots(
        rows=1,
        cols=len(sessions),
        shared_yaxes=True,
        subplot_titles=[f"Session {s}" for s in sessions],
        horizontal_spacing=0.08,
    )

    mats = []
    for sess in sessions:
        ss = sub[sub["session"] == sess].copy()
        ss = ss.sort_values(["target", "trial", "ref_hz"])
        pivot = ss.pivot_table(
            index=["trial", "target"], columns="ref_hz", values="score"
        )
        pivot = pivot.reindex(columns=target_freqs)
        pivot = pivot.sort_index(level=["target", "trial"])
        m = pivot.to_numpy()
        targets = np.array([t for _, t in pivot.index])
        bounds = np.where(np.diff(targets) != 0)[0]
        mats.append((sess, m, targets, bounds))

    zmax = max(m.max() for _, m, _, _ in mats) if mats else 1.0

    for c, (sess, m, targets, bounds) in enumerate(mats, start=1):
        ref_labels = [f"{f} Hz" for f in target_freqs]
        trial_idx = np.arange(m.shape[0])
        fig.add_trace(
            go.Heatmap(
                z=m,
                x=ref_labels,
                y=trial_idx,
                customdata=targets[:, None],
                colorscale="Viridis",
                zmin=0,
                zmax=float(zmax),
                colorbar=dict(title="FBCCA") if c == len(sessions) else None,
                showscale=(c == len(sessions)),
                hovertemplate=(
                    f"Sess {sess}<br>"
                    "trial %{y} • true %{customdata[0]} Hz<br>"
                    "ref %{x}<br>score = %{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=c,
        )
        for b in bounds:
            fig.add_hline(
                y=b + 0.5,
                line=dict(color="white", width=1.2),
                row=1,
                col=c,
            )
        fig.update_xaxes(title_text="Reference", row=1, col=c)

    fig.update_yaxes(
        title_text="Trial (sorted by target)", row=1, col=1, autorange="reversed"
    )

    fig.update_layout(
        title=f"Subject {subject} — FBCCA score matrix (trials × references)",
        height=460,
        margin=dict(l=70, r=20, t=70, b=50),
    )
    return fig
