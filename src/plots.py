"""Figure generation for the SSVEP analysis pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

try:
    from src.analysis import snr_at, welch_psd
    from src.loader import CH_NAMES, TARGET_FREQS, Session
    from src.preprocess import Epochs
except ImportError:
    from analysis import snr_at, welch_psd
    from loader import CH_NAMES, TARGET_FREQS, Session
    from preprocess import Epochs

sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
CLASS_COLORS = {9: "#E63946", 10: "#F4A261", 12: "#2A9D8F", 15: "#264653"}
PLT_DPI = 130


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PLT_DPI, bbox_inches="tight")
    plt.close(fig)


# ----- Per-session figures -------------------------------------------------

def plot_raw_overview(session: Session, eeg_filt: np.ndarray,
                      out: Path, secs: float = 30.0,
                      skip_secs: float = 1.0) -> Path:
    fs = session.sfreq
    skip = int(skip_secs * fs)
    n = int(secs * fs)
    n_end = min(skip + n, eeg_filt.shape[1])
    seg = eeg_filt[:, skip:n_end]
    t = np.arange(seg.shape[1]) / fs + skip_secs

    fig, (ax_t, ax_e) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 6]},
    )
    trig = session.trigger[skip:n_end]
    for f, color in CLASS_COLORS.items():
        mask = trig == f
        ax_t.fill_between(t, 0, 1, where=mask, color=color, alpha=0.85,
                          step="mid", label=f"{f} Hz")
    ax_t.set_yticks([]); ax_t.set_ylabel("Stim", rotation=0, labelpad=20)
    ax_t.legend(ncol=4, loc="upper right", framealpha=0.9, fontsize=10)
    ax_t.set_title(f"{session.title} — Raw signal overview "
                   f"(filtered, t = {skip_secs:.0f}–{skip_secs + secs:.0f} s)")

    # Robust per-channel scale: 95th-percentile abs amplitude
    p95 = np.percentile(np.abs(seg), 95, axis=1)
    spacing = 3 * float(np.median(p95))
    for i, ch in enumerate(CH_NAMES):
        offset = -i * spacing
        ax_e.plot(t, seg[i] + offset, lw=0.7, color="#222")
        ax_e.text(t[0] - 0.4, offset, ch, va="center", ha="right",
                  fontsize=11, fontweight="bold")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_yticks([])
    ax_e.set_xlim(t[0], t[-1])
    ax_e.set_ylim(-len(CH_NAMES) * spacing, spacing)
    p = out / "01_raw_overview.png"
    _save(fig, p)
    return p


def plot_class_distribution(epochs: Epochs, session: Session, out: Path) -> Path:
    counts = {f: int((epochs.labels == f).sum()) for f in (9, 10, 12, 15)}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(k) for k in counts], list(counts.values()),
           color=[CLASS_COLORS[k] for k in counts], edgecolor="#222")
    for k, v in counts.items():
        ax.text(str(k), v + 0.05, str(v), ha="center", fontsize=12,
                fontweight="bold")
    ax.set_xlabel("Stimulation frequency (Hz)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"{session.title} — Trial counts per class")
    ax.set_ylim(0, max(counts.values()) + 2)
    p = out / "02_class_distribution.png"
    _save(fig, p)
    return p


def plot_psd_per_class(epochs: Epochs, session: Session, out: Path) -> Path:
    freqs, psd = welch_psd(epochs)
    occip_idx = [CH_NAMES.index(c) for c in ("O1", "Oz", "O2")]
    keep = freqs <= 35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for f in (9, 10, 12, 15):
        sel = epochs.labels == f
        if not sel.any():
            continue
        avg = psd[sel][:, occip_idx, :].mean(axis=(0, 1))
        ax.plot(freqs[keep], 10 * np.log10(avg[keep] + 1e-30),
                color=CLASS_COLORS[f], lw=2, label=f"{f} Hz target")
    for f, color in CLASS_COLORS.items():
        ax.axvline(f, color=color, ls="--", alpha=0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)  —  averaged over O1/Oz/O2")
    ax.set_title(f"{session.title} — Welch PSD per stimulation class")
    ax.legend(title="Trial class", framealpha=0.9, fontsize=10)
    ax.set_xlim(5, 35)
    p = out / "03_psd_per_class.png"
    _save(fig, p)
    return p


def plot_topomaps(epochs: Epochs, session: Session, out: Path,
                  save: bool = True):
    freqs, psd = welch_psd(epochs)
    info = session.raw.info
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    fig.suptitle(f"{session.title} — Scalp power at each stimulation frequency",
                 fontsize=14)
    cmap = LinearSegmentedColormap.from_list(
        "ssvep", ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E63946"]
    )
    for ax, target in zip(axes, (9, 10, 12, 15)):
        sel = epochs.labels == target
        if not sel.any():
            ax.set_title(f"{target} Hz (no trials)")
            ax.axis("off"); continue
        # mean PSD power (linear) at the target frequency, per channel
        bin_idx = int(np.argmin(np.abs(freqs - target)))
        power = psd[sel, :, bin_idx].mean(axis=0)
        im, _ = mne.viz.plot_topomap(
            power, info, axes=ax, show=False, cmap=cmap,
            sensors=True, contours=4, sphere="auto",
            extrapolate="local",
        )
        ax.set_title(f"{target} Hz")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                 label="PSD at target (µV²/Hz)")
    if save:
        p = out / "04_topomaps.png"
        fig.savefig(p, dpi=PLT_DPI, bbox_inches="tight")
        plt.close(fig)
        return p
    return fig


def plot_snr_heatmap(snr_mat: np.ndarray, session: Session, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(snr_mat, annot=True, fmt=".2f", cmap="rocket_r",
                xticklabels=CH_NAMES,
                yticklabels=[f"{int(t)} Hz" for t in TARGET_FREQS],
                cbar_kws={"label": "SNR (target / sideband)"}, ax=ax)
    ax.set_title(f"{session.title} — SSVEP SNR per channel × class")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Stimulation frequency")
    p = out / "05_snr_heatmap.png"
    _save(fig, p)
    return p


def plot_time_frequency(session: Session, eeg_filt: np.ndarray,
                        out: Path, target: int = 12,
                        chan: str = "Oz", pre: float = 1.0,
                        post: float = 6.0) -> Path:
    from loader import detect_trials
    onsets, _offsets, labels = detect_trials(session.trigger)
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
        # Fallback to next class with trials
        target = int(np.unique(labels)[0])
        return plot_time_frequency(session, eeg_filt, out, target=target,
                                   chan=chan, pre=pre, post=post)

    seg = np.mean(np.stack(segs), axis=0)
    f, t, Sxx = spectrogram(seg, fs=fs, nperseg=fs, noverlap=int(0.9 * fs))
    keep = f <= 35
    t_axis = t - pre

    fig, ax = plt.subplots(figsize=(11, 5))
    pcm = ax.pcolormesh(t_axis, f[keep], 10 * np.log10(Sxx[keep] + 1e-30),
                        cmap="magma", shading="auto")
    ax.axvline(0, color="white", lw=2, ls="--")
    ax.axhline(target, color="white", lw=1, ls=":")
    ax.set_title(f"{session.title} — Time-frequency at {chan}, "
                 f"{target} Hz trials (avg of {len(segs)})")
    ax.set_xlabel("Time relative to stimulus onset (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    p = out / "06_time_frequency.png"
    _save(fig, p)
    return p


def plot_confusion(cm: np.ndarray, classes: list[int], session: Session,
                   out: Path, label: str = "CCA") -> Path:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=[f"{c} Hz" for c in classes],
                yticklabels=[f"{c} Hz" for c in classes],
                vmin=0, vmax=1, cbar_kws={"label": "Row-normalised"}, ax=ax)
    acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
    ax.set_title(f"{session.title} — {label} confusion (acc = {acc:.0%})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    p = out / "07_confusion.png"
    _save(fig, p)
    return p


# ----- Cross-session figures ----------------------------------------------

def plot_snr_comparison(snr_records: list[dict], out: Path) -> Path:
    """snr_records: each dict has keys subject, session, freq, snr."""
    import pandas as pd
    df = pd.DataFrame(snr_records)
    df["who"] = df.apply(lambda r: f"S{r.subject} sess{r.session}", axis=1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=df, x="freq", y="snr", hue="who",
                palette="Set2", ax=ax)
    ax.set_xlabel("Stimulation frequency (Hz)")
    ax.set_ylabel("Mean occipital SNR")
    ax.set_title("SSVEP SNR — subjects × sessions")
    ax.legend(title=None, fontsize=9)
    p = out / "01_snr_comparison.png"
    _save(fig, p)
    return p


def plot_accuracy_curves(rows: list[dict], out: Path) -> Path:
    """rows: each dict {label, windows: dict[float,float], lda_acc}."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = sns.color_palette("Set1", n_colors=len(rows))
    for color, r in zip(palette, rows):
        ws = sorted(r["windows"].keys())
        accs = [r["windows"][w] for w in ws]
        ax.plot(ws, accs, marker="o", lw=2, color=color,
                label=f"{r['label']}  CCA")
        if r.get("lda_acc") is not None:
            ax.axhline(r["lda_acc"], ls=":", lw=1.5, color=color, alpha=0.7)
    ax.axhline(0.25, color="grey", ls="--", lw=1, label="Chance (25%)")
    ax.set_xlabel("Trial window length (s)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_title("CCA decoding accuracy vs. window length\n"
                 "(dotted = online LDA accuracy from CH11)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    p = out / "02_accuracy_vs_window.png"
    _save(fig, p)
    return p


def plot_confusion_grid(cms: list[dict], out: Path) -> Path:
    classes = cms[0]["classes"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, item in zip(axes.flatten(), cms):
        cm = item["cm"]
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=[f"{c}" for c in classes],
                    yticklabels=[f"{c}" for c in classes],
                    vmin=0, vmax=1, cbar=False, ax=ax)
        acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
        ax.set_title(f"{item['label']}  (acc {acc:.0%})", fontsize=12)
        ax.set_xlabel("Predicted (Hz)")
        ax.set_ylabel("True (Hz)")
    fig.suptitle("CCA confusion matrices — all subjects/sessions",
                 fontsize=15, y=1.0)
    p = out / "03_confusion_grid.png"
    _save(fig, p)
    return p
