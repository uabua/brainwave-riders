"""SSVEP analysis: PSD, SNR, CCA-based decoding, time-frequency."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from src.preprocess import Epochs
    from src.loader import detect_trials
except ImportError:
    from preprocess import Epochs
    from loader import detect_trials


def welch_psd(epochs: Epochs, nperseg: int | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, psd) where psd has shape (n_trials, n_channels, n_freqs)."""
    fs = epochs.sfreq
    n_samp = epochs.data.shape[-1]
    if nperseg is None:
        nperseg = min(fs, n_samp)
    freqs, psd = welch(epochs.data, fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, axis=-1)
    return freqs, psd


def snr_at(freqs: np.ndarray, psd: np.ndarray, target: float,
           bw: float = 1.0, gap: float = 0.2) -> np.ndarray:
    """SNR = power at target / mean power in surrounding band, excluding the bin itself.

    Returns an array shaped like psd without its frequency axis.
    """
    target_mask = np.isclose(freqs, target, atol=0.51)
    if not target_mask.any():
        target_idx = int(np.argmin(np.abs(freqs - target)))
        target_mask = np.zeros_like(freqs, dtype=bool)
        target_mask[target_idx] = True
    side_mask = (np.abs(freqs - target) <= bw) & (np.abs(freqs - target) > gap)
    sig = psd[..., target_mask].mean(axis=-1)
    noise = psd[..., side_mask].mean(axis=-1)
    return sig / np.where(noise > 0, noise, np.nan)


def snr_matrix(epochs: Epochs, targets: list[float]
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (snr_per_class, freqs):
       snr_per_class shape = (n_targets, n_channels) — SNR averaged across
       trials of the matching class.
    """
    freqs, psd = welch_psd(epochs)
    n_targets, n_ch = len(targets), psd.shape[1]
    out = np.zeros((n_targets, n_ch))
    for i, t in enumerate(targets):
        mask = epochs.labels == int(t)
        if not mask.any():
            out[i] = np.nan
            continue
        snr = snr_at(freqs, psd[mask], t)  # (n_match_trials, n_ch)
        out[i] = snr.mean(axis=0)
    return out, freqs


# ----------------- CCA-based decoding -----------------

def cca_references(target: float, n_samples: int, sfreq: int,
                   n_harmonics: int = 2) -> np.ndarray:
    t = np.arange(n_samples) / sfreq
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * target * t))
        refs.append(np.cos(2 * np.pi * h * target * t))
    return np.array(refs).T  # (n_samples, 2*n_harmonics)


def cca_score(trial: np.ndarray, target: float, sfreq: int,
              n_harmonics: int = 2) -> float:
    """First canonical correlation between trial and reference signals."""
    X = trial.T  # (n_samples, n_channels)
    Y = cca_references(target, X.shape[0], sfreq, n_harmonics)
    cca = CCA(n_components=1, max_iter=500)
    Xc, Yc = cca.fit_transform(X, Y)
    Xc = Xc.flatten(); Yc = Yc.flatten()
    if Xc.std() == 0 or Yc.std() == 0:
        return 0.0
    return float(np.corrcoef(Xc, Yc)[0, 1])


def cca_decode(epochs: Epochs, targets: list[float], n_harmonics: int = 2
               ) -> tuple[np.ndarray, np.ndarray]:
    """Predict label of each epoch by argmax CCA correlation across targets.

    Returns (preds, scores) where scores shape = (n_trials, n_targets).
    """
    n_trials = epochs.data.shape[0]
    scores = np.zeros((n_trials, len(targets)))
    for i in range(n_trials):
        trial = epochs.data[i]
        for j, t in enumerate(targets):
            scores[i, j] = cca_score(trial, t, epochs.sfreq, n_harmonics)
    preds_idx = scores.argmax(axis=1)
    preds = np.array([targets[k] for k in preds_idx], dtype=int)
    return preds, scores


def confusion(true: np.ndarray, pred: np.ndarray, classes: list[int]
              ) -> np.ndarray:
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true, pred):
        if t in classes and p in classes:
            cm[classes.index(int(t)), classes.index(int(p))] += 1
    return cm


def accuracy_vs_window(epochs_full: Epochs, targets: list[float],
                       windows_sec: list[float]) -> dict[float, float]:
    """For each window length, truncate epochs to (0..window) and decode."""
    out = {}
    for w in windows_sec:
        n = int(round(w * epochs_full.sfreq))
        n = min(n, epochs_full.data.shape[-1])
        truncated = Epochs(
            data=epochs_full.data[..., :n],
            labels=epochs_full.labels,
            sfreq=epochs_full.sfreq,
            tmin=epochs_full.tmin,
            tmax=epochs_full.tmin + n / epochs_full.sfreq,
            ch_names=epochs_full.ch_names,
        )
        preds, _ = cca_decode(truncated, targets)
        out[w] = float((preds == truncated.labels).mean())
    return out


# ----------------- Feature extraction & dimensionality reduction -----------

def psd_features(epochs: Epochs, ch_indices: list[int],
                 ch_names: list[str] | None = None,
                 fmin: float = 5.0, fmax: float = 35.0) -> pd.DataFrame:
    """Per-trial PSD vector (selected channels × freq bins in [fmin, fmax])."""
    freqs, psd = welch_psd(epochs)
    keep = (freqs >= fmin) & (freqs <= fmax)
    feat = psd[:, ch_indices, :][:, :, keep]
    n_trials = feat.shape[0]
    flat = feat.reshape(n_trials, -1)
    kept_freqs = freqs[keep]
    ch_lbls = ([ch_names[i] for i in ch_indices] if ch_names
               else [f"ch{i}" for i in ch_indices])
    cols = [f"{lbl}_{kf:.0f}Hz" for lbl in ch_lbls for kf in kept_freqs]
    df = pd.DataFrame(flat, columns=cols)
    df.insert(0, "trial", np.arange(n_trials))
    df.insert(1, "target", epochs.labels.astype(int))
    return df


def cca_features(scores: np.ndarray, labels: np.ndarray,
                 targets: list[float]) -> pd.DataFrame:
    """Per-trial CCA correlation vector (one column per target frequency)."""
    cols = [f"cca_{int(t)}Hz" for t in targets]
    df = pd.DataFrame(scores, columns=cols)
    df.insert(0, "trial", np.arange(scores.shape[0]))
    df.insert(1, "target", labels.astype(int))
    return df


def project(feat_df: pd.DataFrame, method: str = "pca",
            n_components: int = 3, random_state: int = 42
            ) -> tuple[np.ndarray, list[str]]:
    """Standardize then reduce. Returns (coords, axis_labels)."""
    drop_cols = [c for c in ("trial", "target", "session") if c in feat_df.columns]
    X = feat_df.drop(columns=drop_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    if method == "pca":
        red = PCA(n_components=n_components)
        coords = red.fit_transform(X_scaled)
        labels = [f"PC{i + 1} ({red.explained_variance_ratio_[i]:.1%})"
                  for i in range(n_components)]
    elif method == "umap":
        import umap as umap_lib
        n_samples = X_scaled.shape[0]
        red = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=min(5, max(2, n_samples - 1)),
            min_dist=0.3,
            random_state=random_state,
        )
        coords = red.fit_transform(X_scaled)
        labels = [f"UMAP{i + 1}" for i in range(n_components)]
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return coords, labels


def online_lda_accuracy(session, lda_to_freq: dict[int, float | None]
                        ) -> tuple[float, np.ndarray, np.ndarray]:
    """Use the online LDA channel: decision per trial = mode of LDA during stim."""
    onsets, offsets, labels = detect_trials(session.trigger)
    correct = 0
    truth, pred = [], []
    for on, off, lbl in zip(onsets, offsets, labels):
        seg = session.lda[on:off]
        nz = seg[seg != 0]
        if len(nz) == 0:
            decision = 0
        else:
            uniq, cnt = np.unique(nz, return_counts=True)
            decision = int(uniq[np.argmax(cnt)])
        decoded_freq = lda_to_freq.get(decision, None)
        truth.append(int(lbl))
        pred.append(int(decoded_freq) if decoded_freq is not None else 0)
        if decoded_freq is not None and int(decoded_freq) == int(lbl):
            correct += 1
    acc = correct / max(1, len(onsets))
    return acc, np.array(truth), np.array(pred)
