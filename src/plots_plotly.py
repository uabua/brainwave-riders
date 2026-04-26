"""Interactive Plotly versions of the SSVEP plots in plots.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

try:
    from src.analysis import snr_at, welch_psd
    from src.loader import CH_NAMES, TARGET_FREQS, Session, detect_trials
    from src.preprocess import Epochs
except ImportError:
    from analysis import snr_at, welch_psd
    from loader import CH_NAMES, TARGET_FREQS, Session, detect_trials
    from preprocess import Epochs

# Aligned to app.py story palette (ACCENT_2 / ACCENT_3 / ACCENT_4 / ACCENT)
CLASS_COLORS = {9: "#3f6fd8", 10: "#1f8a70", 12: "#af7d2d", 15: "#d95d39"}
SESSION_SYMBOLS = {1: "circle", 2: "square", 3: "diamond", 4: "cross"}

# ── Theme helper — transparent bg so charts breathe with the parchment app ──
_PANEL = "#fffdf9"

def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_PANEL,
        font=dict(color="#181715", family="system-ui, -apple-system, sans-serif"),
    )
    return fig




def raw_overview(session: Session, eeg_filt: np.ndarray,
                 secs: float | None = None, skip_secs: float | None = None,
                 pad: float = 1.0) -> go.Figure:
    fs = session.sfreq
    if skip_secs is None or secs is None:
        onsets, offsets, labels = detect_trials(session.trigger)
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
        if start_i is not None:
            skip_secs = max(0.0, onsets[start_i] / fs - pad)
            secs = (offsets[end_i] / fs + pad) - skip_secs
        else:
            skip_secs, secs = 1.0, 30.0
    skip = int(skip_secs * fs)
    n = int(secs * fs)
    n_end = min(skip + n, eeg_filt.shape[1])
    seg = eeg_filt[:, skip:n_end]
    t = np.arange(seg.shape[1]) / fs + skip_secs
    trig = session.trigger[skip:n_end]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.2, 0.8], vertical_spacing=0.09,
                        subplot_titles=("Stimulation", "EEG"))

    # Stim band: one filled trace per class
    for f, color in CLASS_COLORS.items():
        mask = (trig == f).astype(float)
        fig.add_trace(go.Scatter(
            x=t, y=mask, fill="tozeroy", mode="none",
            fillcolor=color, opacity=0.85, name=f"{f} Hz",
            hovertemplate=f"%{{x:.2f}}s  —  {f} Hz<extra></extra>",
        ), row=1, col=1)

    # EEG: stack channels
    p95 = np.percentile(np.abs(seg), 95, axis=1)
    spacing = 3 * float(np.median(p95))
    for i, ch in enumerate(CH_NAMES):
        offset = -i * spacing
        fig.add_trace(go.Scatter(
            x=t, y=seg[i] + offset, mode="lines",
            line=dict(color="#222", width=0.8),
            name=ch, showlegend=False,
            hovertemplate=f"{ch}<br>%{{x:.2f}}s<br>%{{y:.1f}} µV<extra></extra>",
        ), row=2, col=1)

    fig.update_yaxes(showticklabels=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(
        showticklabels=True, row=2, col=1,
        tickmode="array",
        tickvals=[-i * spacing for i in range(len(CH_NAMES))],
        ticktext=CH_NAMES,
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        title=f"{session.title} — Raw signal overview "
              f"(filtered, t = {skip_secs:.0f}–{skip_secs + secs:.0f} s)",
        height=600, margin=dict(l=60, r=20, t=100, b=50),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return _apply_theme(fig)


def class_distribution(epochs: Epochs, session: Session) -> go.Figure:
    counts = {f: int((epochs.labels == f).sum()) for f in (9, 10, 12, 15)}
    fig = go.Figure(go.Bar(
        x=[f"{k} Hz" for k in counts],
        y=list(counts.values()),
        marker_color=[CLASS_COLORS[k] for k in counts],
        marker_line_color="#222", marker_line_width=1,
        text=[str(v) for v in counts.values()],
        textposition="outside", textfont=dict(size=14),
        hovertemplate="%{x}<br>%{y} trials<extra></extra>",
    ))
    fig.update_layout(
        title=f"{session.title} — Trial counts per class",
        xaxis_title="Stimulation frequency",
        yaxis_title="Number of trials",
        yaxis=dict(range=[0, max(counts.values()) + 2]),
        height=420, margin=dict(l=60, r=20, t=70, b=50),
    )
    return _apply_theme(fig)


def psd_per_class(epochs: Epochs, session: Session) -> go.Figure:
    freqs, psd = welch_psd(epochs)
    occip_idx = [CH_NAMES.index(c) for c in ("O1", "Oz", "O2")]
    keep = freqs <= 35

    fig = go.Figure()
    for f in (9, 10, 12, 15):
        sel = epochs.labels == f
        if not sel.any():
            continue
        avg = psd[sel][:, occip_idx, :].mean(axis=(0, 1))
        fig.add_trace(go.Scatter(
            x=freqs[keep], y=10 * np.log10(avg[keep] + 1e-30),
            mode="lines", name=f"{f} Hz target",
            line=dict(color=CLASS_COLORS[f], width=2.5),
            hovertemplate=f"{f} Hz target<br>%{{x:.1f}} Hz<br>%{{y:.1f}} dB"
                          "<extra></extra>",
        ))
    for f, color in CLASS_COLORS.items():
        fig.add_vline(x=f, line=dict(color=color, dash="dash", width=1),
                      opacity=0.4)
    fig.update_layout(
        title=f"{session.title} — Welch PSD per stimulation class",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB) — averaged over O1/Oz/O2",
        xaxis=dict(range=[5, 35]),
        legend=dict(title="Trial class"),
        height=480, margin=dict(l=60, r=20, t=70, b=50),
    )
    return _apply_theme(fig)


def snr_heatmap(snr_mat: np.ndarray, session: Session) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=snr_mat,
        x=CH_NAMES,
        y=[f"{int(t)} Hz" for t in TARGET_FREQS],
        text=snr_mat,
        texttemplate="%{text:.2f}",
        textfont=dict(size=12),
        colorscale="Plasma_r",
        colorbar=dict(title="SNR"),
        hovertemplate="Channel %{x}<br>%{y}<br>SNR %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{session.title} — SSVEP SNR per channel × class",
        xaxis_title="Channel",
        yaxis_title="Stimulation frequency",
        yaxis_autorange="reversed",
        height=420, margin=dict(l=80, r=20, t=70, b=60),
    )
    return _apply_theme(fig)


def psd_per_channel(epochs: Epochs, session: Session,
                    fmin: float = 5.0, fmax: float = 50.0) -> go.Figure:
    """2×4 grid of PSD curves — one panel per electrode, classes overlaid."""
    freqs, psd = welch_psd(epochs)
    keep = (freqs >= fmin) & (freqs <= fmax)
    fig = make_subplots(rows=2, cols=4, subplot_titles=list(CH_NAMES),
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.04, vertical_spacing=0.14)
    for ch_idx, ch_name in enumerate(CH_NAMES):
        r, c = ch_idx // 4 + 1, ch_idx % 4 + 1
        for f in (9, 10, 12, 15):
            sel = epochs.labels == f
            if not sel.any():
                continue
            avg = psd[sel, ch_idx, :][:, keep].mean(axis=0)
            fig.add_trace(go.Scatter(
                x=freqs[keep], y=avg, mode="lines",
                name=f"{f} Hz",
                line=dict(color=CLASS_COLORS[f], width=1.5),
                showlegend=(ch_idx == 0),
                legendgroup=f"{f}",
                hovertemplate=f"{ch_name} · {f} Hz<br>%{{x:.1f}} Hz<br>"
                              "%{y:.2g}<extra></extra>",
            ), row=r, col=c)
        fig.update_xaxes(range=[fmin, fmax], row=r, col=c)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2)
    fig.update_yaxes(title_text="Power", col=1)
    fig.update_layout(
        title=f"{session.title} — PSD per channel (classes overlaid)",
        height=480, margin=dict(l=50, r=20, t=80, b=60),
        legend=dict(title="Class", orientation="h", y=-0.05),
    )
    return _apply_theme(fig)


def rest_vs_stimulus(session: Session, eeg_filt: np.ndarray,
                     fmin: float = 5.0, fmax: float = 50.0) -> go.Figure:
    """Mean-channel PSD during rest vs stimulation, plus % evoked change."""
    fs = session.sfreq
    trig = session.trigger
    rest_mask = trig == 0
    stim_mask = trig != 0
    nperseg = min(1024, int(rest_mask.sum()), int(stim_mask.sum()))
    rest_acc, stim_acc, freqs = [], [], None
    for ch in range(eeg_filt.shape[0]):
        freqs, rp = welch(eeg_filt[ch, rest_mask], fs=fs,
                          nperseg=nperseg, noverlap=nperseg // 2)
        _, sp = welch(eeg_filt[ch, stim_mask], fs=fs,
                      nperseg=nperseg, noverlap=nperseg // 2)
        rest_acc.append(rp); stim_acc.append(sp)
    rest_avg = np.mean(rest_acc, axis=0)
    stim_avg = np.mean(stim_acc, axis=0)
    contrast = (stim_avg - rest_avg) / (rest_avg + 1e-10) * 100
    keep = (freqs >= fmin) & (freqs <= fmax)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=("Rest vs Stimulus PSD", "Evoked response (% change)"),
    )
    fig.add_trace(go.Scatter(
        x=freqs[keep], y=rest_avg[keep], mode="lines", name="Rest",
        line=dict(color="#888", width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=freqs[keep], y=stim_avg[keep], mode="lines", name="Stimulus",
        line=dict(color="#1f77b4", width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=freqs[keep], y=contrast[keep], mode="lines",
        line=dict(color="#2ca02c", width=1.8), showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="#444", width=0.5), row=1, col=2)
    for sf, color in CLASS_COLORS.items():
        fig.add_vline(x=sf, line=dict(color=color, dash="dash", width=1),
                      opacity=0.4, row=1, col=1)
        fig.add_vline(x=sf, line=dict(color=color, dash="dash", width=1),
                      opacity=0.4, row=1, col=2)
    fig.update_yaxes(type="log", title_text="PSD", row=1, col=1)
    fig.update_yaxes(title_text="% change", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[fmin, fmax], row=1)
    fig.update_layout(
        title=f"{session.title} — Rest vs Stimulus",
        height=420, margin=dict(l=60, r=20, t=70, b=50),
        legend=dict(orientation="h", y=-0.15),
    )
    return _apply_theme(fig)


def snr_class_heatmap(epochs: Epochs, session: Session,
                      targets: tuple[int, ...] = (9, 10, 12, 15),
                      ch_indices: list[int] | None = None) -> go.Figure:
    """4×4 heatmap: mean SNR per (true class, stimulation reference freq)."""
    freqs, psd = welch_psd(epochs)
    if ch_indices is None:
        ch_indices = list(range(psd.shape[1]))
    n = len(targets)
    mat = np.full((n, n), np.nan)
    for i, t in enumerate(targets):
        sel = epochs.labels == int(t)
        if not sel.any():
            continue
        for j, sf in enumerate(targets):
            snr = snr_at(freqs, psd[sel][:, ch_indices, :], float(sf))
            mat[i, j] = float(np.nanmean(snr))
    fig = go.Figure(go.Heatmap(
        z=mat,
        x=[f"{t} Hz" for t in targets],
        y=[f"{t} Hz" for t in targets],
        text=mat, texttemplate="%{text:.2f}", textfont=dict(size=14),
        colorscale="YlOrRd",
        colorbar=dict(title="SNR"),
        hovertemplate="True class %{y}<br>Reference %{x}<br>"
                      "SNR %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{session.title} — SNR (true class × ref freq)",
        xaxis=dict(title="Reference frequency", constrain="domain"),
        yaxis=dict(title="True class", autorange="reversed",
                   scaleanchor="x", constrain="domain"),
        height=420, margin=dict(l=80, r=20, t=70, b=60),
    )
    return _apply_theme(fig)


def cca_correlation_heatmap(scores: np.ndarray, labels: np.ndarray,
                            session: Session,
                            targets: tuple[int, ...] = (9, 10, 12, 15)
                            ) -> go.Figure:
    """4×4 heatmap: mean CCA correlation per (true class, ref freq)."""
    n = len(targets)
    mat = np.full((n, n), np.nan)
    labels_int = labels.astype(int)
    for i, t in enumerate(targets):
        sel = labels_int == int(t)
        if not sel.any():
            continue
        mat[i] = scores[sel].mean(axis=0)
    fig = go.Figure(go.Heatmap(
        z=mat,
        x=[f"{t} Hz" for t in targets],
        y=[f"{t} Hz" for t in targets],
        text=mat, texttemplate="%{text:.3f}", textfont=dict(size=13),
        colorscale="YlOrRd",
        colorbar=dict(title="Correlation"),
        hovertemplate="True class %{y}<br>Reference %{x}<br>"
                      "r %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{session.title} — CCA correlation (class × ref)",
        xaxis=dict(title="Reference frequency", constrain="domain"),
        yaxis=dict(title="True class", autorange="reversed",
                   scaleanchor="x", constrain="domain"),
        height=420, margin=dict(l=80, r=20, t=70, b=60),
    )
    return _apply_theme(fig)


def time_frequency(session: Session, eeg_filt: np.ndarray,
                   target: int = 12, chan: str = "Oz",
                   pre: float = 1.0, post: float = 6.0) -> go.Figure:
    onsets, _, labels = detect_trials(session.trigger)
    fs = session.sfreq
    ch_idx = CH_NAMES.index(chan)
    pre_n, post_n = int(pre * fs), int(post * fs)

    segs = []
    for on, lbl in zip(onsets, labels):
        if int(lbl) != target:
            continue
        s, e = on - pre_n, on + post_n
        if s < 0 or e > eeg_filt.shape[1]:
            continue
        segs.append(eeg_filt[ch_idx, s:e])
    if not segs:
        target = int(np.unique(labels)[0])
        return time_frequency(session, eeg_filt, target=target,
                              chan=chan, pre=pre, post=post)

    seg = np.mean(np.stack(segs), axis=0)
    f, t, Sxx = spectrogram(seg, fs=fs, nperseg=fs, noverlap=int(0.9 * fs))
    keep = f <= 35
    t_axis = t - pre
    z = 10 * np.log10(Sxx[keep] + 1e-30)

    fig = go.Figure(go.Heatmap(
        x=t_axis, y=f[keep], z=z, colorscale="Magma",
        colorbar=dict(title="Power (dB)"),
        hovertemplate="t = %{x:.2f} s<br>%{y:.1f} Hz<br>%{z:.1f} dB"
                      "<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="white", width=2, dash="dash"))
    fig.add_hline(y=target, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(
        title=f"{session.title} — Time-frequency at {chan}, "
              f"{target} Hz trials (avg of {len(segs)})",
        xaxis_title="Time relative to stimulus onset (s)",
        yaxis_title="Frequency (Hz)",
        height=480, margin=dict(l=60, r=20, t=70, b=50),
    )
    return _apply_theme(fig)


def confusion(cm: np.ndarray, classes: list[int], session: Session,
              label: str = "CCA") -> go.Figure:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=[f"{c} Hz" for c in classes],
        y=[f"{c} Hz" for c in classes],
        text=cm, texttemplate="%{text:d}", textfont=dict(size=14),
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="Row-normalised"),
        hovertemplate="True %{y}<br>Pred %{x}<br>"
                      "%{text:d} trials (%{z:.0%})<extra></extra>",
    ))
    fig.update_layout(
        title=f"{session.title} — {label} confusion (acc = {acc:.0%})",
        xaxis=dict(title="Predicted", constrain="domain"),
        yaxis=dict(title="True", autorange="reversed",
                   scaleanchor="x", constrain="domain"),
        height=480, margin=dict(l=80, r=20, t=70, b=60),
    )
    return _apply_theme(fig)


# ----- Cross-session figures ----------------------------------------------

def snr_comparison(snr_records: list[dict]) -> go.Figure:
    df = pd.DataFrame(snr_records)
    df["who"] = df.apply(lambda r: f"Sub {int(r.subject)} · sess {int(r.session)}", axis=1)
    # Subject-session colours aligned to app palette
    _SNR_PAL = ["#3f6fd8", "#7fa0e8", "#d95d39", "#e8997d"]  # S1s1,S1s2,S2s1,S2s2
    fig = px.bar(df, x="freq", y="snr", color="who", barmode="group",
                 color_discrete_sequence=_SNR_PAL,
                 labels={"freq": "Stimulation frequency (Hz)",
                         "snr": "Mean occipital SNR",
                         "who": ""})
    fig.update_layout(
        title="SSVEP SNR — subjects × sessions",
        xaxis=dict(type="category"),
        height=480, margin=dict(l=60, r=20, t=70, b=50),
    )
    return _apply_theme(fig)


def accuracy_curves(rows: list[dict]) -> go.Figure:
    # Two subjects × two sessions → blue shades / orange shades
    palette = ["#3f6fd8", "#7fa0e8", "#d95d39", "#e8997d"]
    fig = go.Figure()
    for color, r in zip(palette, rows):
        ws = sorted(r["windows"].keys())
        accs = [r["windows"][w] for w in ws]
        fig.add_trace(go.Scatter(
            x=ws, y=accs, mode="lines+markers", name=f"{r['label']} CCA",
            line=dict(color=color, width=2.5),
            marker=dict(size=9),
            hovertemplate=f"{r['label']}<br>%{{x:.0f}} s window<br>"
                          "%{y:.0%}<extra></extra>",
        ))
        if r.get("lda_acc") is not None:
            fig.add_hline(
                y=r["lda_acc"],
                line=dict(color=color, dash="dot", width=1.5),
                opacity=0.7,
                annotation_text=f"{r['label']} online LDA = {r['lda_acc']:.0%}",
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
            )
    fig.add_hline(y=0.25, line=dict(color="grey", dash="dash", width=1),
                  annotation_text="Chance (25%)", annotation_position="left",
                  annotation_font=dict(size=10, color="grey"))
    fig.update_layout(
        title="CCA decoding accuracy vs. window length<br>"
              "<sup>(dotted = online LDA accuracy from CH11)</sup>",
        xaxis_title="Trial window length (s)",
        yaxis_title="Decoding accuracy",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=520, margin=dict(l=60, r=160, t=80, b=50),
    )
    return _apply_theme(fig)


def feature_space_3d(coords: np.ndarray, axis_labels: list[str],
                     targets: np.ndarray, title: str) -> go.Figure:
    """3D scatter of trials, color = target class."""
    targets_arr = np.asarray(targets, dtype=int)
    fig = go.Figure()
    for t in sorted(set(targets_arr.tolist())):
        mask = targets_arr == t
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers", name=f"{t} Hz",
            marker=dict(size=5, color=CLASS_COLORS.get(t, "#888"),
                        opacity=0.85, line=dict(width=0)),
            hovertemplate=f"{t} Hz<br>%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}"
                          "<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=axis_labels[0],
                   yaxis_title=axis_labels[1],
                   zaxis_title=axis_labels[2]),
        height=480, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(title="Class", orientation="h", y=-0.05),
    )
    return _apply_theme(fig)


def feature_space_3d_overlay(coords: np.ndarray, axis_labels: list[str],
                             targets: np.ndarray, sessions: np.ndarray,
                             title: str) -> go.Figure:
    """3D scatter, color = class, marker symbol = session."""
    targets_arr = np.asarray(targets, dtype=int)
    sessions_arr = np.asarray(sessions, dtype=int)
    fig = go.Figure()
    for t in sorted(set(targets_arr.tolist())):
        for s in sorted(set(sessions_arr.tolist())):
            mask = (targets_arr == t) & (sessions_arr == s)
            if not mask.any():
                continue
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers", name=f"{t} Hz · S{s}",
                marker=dict(size=5,
                            color=CLASS_COLORS.get(t, "#888"),
                            symbol=SESSION_SYMBOLS.get(s, "circle"),
                            opacity=0.85, line=dict(width=0)),
                hovertemplate=f"{t} Hz · session {s}<br>"
                              "%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>",
            ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=axis_labels[0],
                   yaxis_title=axis_labels[1],
                   zaxis_title=axis_labels[2]),
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.05),
    )
    return _apply_theme(fig)


def confusion_grid(cms: list[dict]) -> go.Figure:
    classes = cms[0]["classes"]
    # Short titles to prevent overlap — acc on second line via <br>
    titles = [
        f"{item['label']}<br><sup>acc {np.trace(item['cm']) / item['cm'].sum():.0%}</sup>"
        for item in cms
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.18,   # more room so titles don't bleed across
        vertical_spacing=0.20,     # more room for the two-line titles
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
    )
    for idx, item in enumerate(cms):
        r, c = idx // 2 + 1, idx % 2 + 1
        cm = item["cm"]
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig.add_trace(go.Heatmap(
            z=cm_norm, x=[f"{cl} Hz" for cl in classes],
            y=[f"{cl} Hz" for cl in classes],
            text=cm, texttemplate="%{text:d}", textfont=dict(size=12),
            colorscale="Blues", zmin=0, zmax=1, showscale=(idx == 0),
            colorbar=dict(title="Row-norm", len=0.42, y=0.78) if idx == 0 else None,
            hovertemplate=f"{item['label']}<br>True %{{y}}<br>"
                          "Pred %{x}<br>%{text:d} trials<extra></extra>",
        ), row=r, col=c)
        # Only label left column to avoid the diagonal-label bleed on right panels
        fig.update_xaxes(title_text="Predicted", constrain="domain", row=r, col=c)
        fig.update_yaxes(
            title_text="True" if c == 1 else "",
            autorange="reversed",
            constrain="domain",
            row=r, col=c,
        )
    fig.update_layout(
        title="CCA confusion matrices — all subjects/sessions",
        height=800,
        margin=dict(l=70, r=40, t=100, b=50),
    )
    return _apply_theme(fig)
