from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import CH_NAMES, CLASS_COLORS, FS


def _auto_window(stim: np.ndarray, fs: int, pad: float) -> tuple[float, float]:
    """Find the shortest contiguous span of trials covering all target classes."""
    on = stim > 0
    onsets = np.where(np.diff(on.astype(np.int8)) == 1)[0] + 1
    offsets = np.where(np.diff(on.astype(np.int8)) == -1)[0] + 1
    if len(onsets) == 0 or len(offsets) == 0:
        return 1.0, 30.0
    if offsets[0] < onsets[0]:
        offsets = offsets[1:]
    n = min(len(onsets), len(offsets))
    onsets, offsets = onsets[:n], offsets[:n]
    labels = stim[onsets]
    needed = set(CLASS_COLORS)

    start_i = end_i = None
    for i in range(len(onsets)):
        seen: set[int] = set()
        for j in range(i, len(onsets)):
            seen.add(int(labels[j]))
            if seen >= needed:
                start_i, end_i = i, j
                break
        if start_i is not None:
            break

    if start_i is None:
        return 1.0, 30.0

    skip = max(0.0, onsets[start_i] / fs - pad)
    secs = (offsets[end_i] / fs + pad) - skip
    return skip, secs


def raw_overview(
    df: pd.DataFrame,
    title: str,
    *,
    secs: float | None = None,
    skip_secs: float | None = None,
    pad: float = 1.0,
    fs: int = FS,
    ch_names: list[str] = CH_NAMES,
) -> go.Figure:
    stim = df["stimulus"].to_numpy()

    if skip_secs is None or secs is None:
        skip_secs, secs = _auto_window(stim, fs, pad)

    skip = int(skip_secs * fs)
    n = int(secs * fs)
    end = min(skip + n, len(df))

    seg = df.iloc[skip:end]
    t = np.arange(end - skip) / fs + skip_secs
    trig = stim[skip:end]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.2, 0.8],
        vertical_spacing=0.09,
        subplot_titles=("Stimulation", "EEG"),
    )

    for f, color in CLASS_COLORS.items():
        mask = (trig == f).astype(float)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=mask,
                fill="tozeroy",
                mode="none",
                fillcolor=color,
                opacity=0.85,
                name=f"{f} Hz",
                hovertemplate=f"%{{x:.2f}}s  —  {f} Hz<extra></extra>",
            ),
            row=1,
            col=1,
        )

    eeg = seg[ch_names].to_numpy().T
    p95 = np.percentile(np.abs(eeg), 95, axis=1)
    spacing = 3 * float(np.median(p95))
    for i, ch in enumerate(ch_names):
        offset = -i * spacing
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=eeg[i] + offset,
                mode="lines",
                line=dict(color="#222", width=0.8),
                name=ch,
                showlegend=False,
                hovertemplate=f"{ch}<br>%{{x:.2f}}s<br>%{{y:.1f}} µV<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(showticklabels=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(
        showticklabels=True,
        row=2,
        col=1,
        tickmode="array",
        tickvals=[-i * spacing for i in range(len(ch_names))],
        ticktext=ch_names,
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        title=f"{title} — Raw signal overview "
        f"(filtered, t = {skip_secs:.0f}–{skip_secs + secs:.0f} s)",
        height=600,
        margin=dict(l=60, r=20, t=100, b=50),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return fig
