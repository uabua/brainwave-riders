from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import TARGET_FREQS


def _trial_x_ref_matrix(
    cca_df: pd.DataFrame,
    subject: int,
    session: int,
    *,
    n_harmonics: int,
    target_freqs: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (matrix, target_per_trial, group_boundaries) sorted by target."""
    sub = cca_df[(cca_df["subject"] == subject) & (cca_df["session"] == session)].copy()
    sub = sub.sort_values(["target", "trial"]).reset_index(drop=True)
    cols = [f"cca_{f}Hz_h{n_harmonics}" for f in target_freqs]
    matrix = sub[cols].to_numpy()
    targets = sub["target"].to_numpy()
    boundaries = np.where(np.diff(targets) != 0)[0]
    return matrix, targets, boundaries


def cca_matrix(
    cca_df: pd.DataFrame,
    subject: int,
    *,
    n_harmonics: int = 3,
    target_freqs: list[int] = TARGET_FREQS,
) -> go.Figure:
    """Trial × reference-frequency CCA score heatmap, trials sorted by true target."""
    sub = cca_df[cca_df["subject"] == subject]
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
        m, t, b = _trial_x_ref_matrix(
            cca_df,
            subject,
            sess,
            n_harmonics=n_harmonics,
            target_freqs=target_freqs,
        )
        mats.append((sess, m, t, b))

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
                colorbar=dict(title="CCA r") if c == len(sessions) else None,
                showscale=(c == len(sessions)),
                hovertemplate=(
                    f"Sess {sess}<br>"
                    "trial %{y} • true %{customdata[0]} Hz<br>"
                    "ref %{x}<br>r = %{z:.3f}<extra></extra>"
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
        title=(
            f"Subject {subject} — CCA correlation matrix "
            f"(trials × references, n_harmonics={n_harmonics})"
        ),
        height=460,
        margin=dict(l=70, r=20, t=70, b=50),
    )
    return fig
