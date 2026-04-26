from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy import signal


def extract_features_psd(
    df: pd.DataFrame,
    eeg_cols: list[str],
    stim_freqs: list[int],
    pre_sec: float = 0.5,
    win_sec: float = 6.85,
    fs: int = 256,
) -> pd.DataFrame:
    pre_samples = int(pre_sec * fs)
    win_samples = int(win_sec * fs)
    features_list = []

    n_trials = int(df["trial"].max())
    for trial_num in range(1, n_trials + 1):
        trial_data = df[df["trial"] == trial_num]
        target = int(trial_data["stimulus"].iloc[0])
        eeg = trial_data[eeg_cols].to_numpy()

        segment = eeg[pre_samples : pre_samples + win_samples, :]
        if segment.shape[0] < win_samples:
            continue

        feat = {"trial": trial_num, "target": target}
        nperseg = min(1024, len(segment))

        for ch in range(len(eeg_cols)):
            ch_data = segment[:, ch]
            f, psd = signal.welch(
                ch_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
            )

            for sf in stim_freqs:
                for h in range(1, 4):
                    freq = sf * h
                    if freq < fs / 2:
                        idx = int(np.argmin(np.abs(f - freq)))
                        feat[f"ch{ch+1}_psd_{sf}Hz_h{h}"] = float(psd[idx])
                        neighbors = list(range(max(0, idx - 5), idx - 1)) + list(
                            range(idx + 2, min(len(psd), idx + 6))
                        )
                        if neighbors:
                            feat[f"ch{ch+1}_snr_{sf}Hz_h{h}"] = float(
                                psd[idx] / (np.mean(psd[neighbors]) + 1e-10)
                            )

            powers = {
                sf: float(psd[int(np.argmin(np.abs(f - sf)))]) for sf in stim_freqs
            }
            total = sum(powers.values())
            for sf in stim_freqs:
                feat[f"ch{ch+1}_relpower_{sf}Hz"] = powers[sf] / (total + 1e-10)

            for band, (lo, hi) in {
                "theta": (4, 8),
                "alpha": (8, 13),
                "low_beta": (13, 20),
                "high_beta": (20, 30),
            }.items():
                mask = (f >= lo) & (f <= hi)
                feat[f"ch{ch+1}_{band}"] = float(np.mean(psd[mask]))

            feat[f"ch{ch+1}_var"] = float(np.var(ch_data))

        features_list.append(feat)

    return pd.DataFrame(features_list)


def create_reference(
    freq: float, fs: int, n_samples: int, n_harmonics: int = 3
) -> np.ndarray:
    t = np.arange(n_samples) / fs
    ref = []
    for h in range(1, n_harmonics + 1):
        ref.append(np.sin(2 * np.pi * h * freq * t))
        ref.append(np.cos(2 * np.pi * h * freq * t))
    return np.stack(ref, axis=1)


def cca_score(eeg: np.ndarray, ref: np.ndarray) -> float:
    x = eeg - eeg.mean(axis=0, keepdims=True)
    y = ref - ref.mean(axis=0, keepdims=True)
    ux, sx, _ = svd(x, full_matrices=False)
    uy, sy, _ = svd(y, full_matrices=False)
    tol_x = max(x.shape) * np.finfo(float).eps * sx[0]
    tol_y = max(y.shape) * np.finfo(float).eps * sy[0]
    qx = ux[:, : int(np.sum(sx > tol_x))]
    qy = uy[:, : int(np.sum(sy > tol_y))]
    corr = svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(corr[0], 0.0, 1.0)) if corr.size else 0.0


def extract_features_cca(
    df: pd.DataFrame,
    eeg_cols: list[str],
    stim_freqs: list[int],
    harmonics_list: list[int],
    pre_sec: float = 0.5,
    win_sec: float = 6.85,
    fs: int = 256,
) -> pd.DataFrame:
    pre_samples = int(pre_sec * fs)
    win_samples = int(win_sec * fs)
    features_list = []

    n_trials = int(df["trial"].max())
    for trial_num in range(1, n_trials + 1):
        trial_data = df[df["trial"] == trial_num]
        target = int(trial_data["stimulus"].iloc[0])
        eeg = trial_data[eeg_cols].to_numpy()

        segment = eeg[pre_samples : pre_samples + win_samples, :]
        if segment.shape[0] < win_samples:
            continue

        feat = {"trial": trial_num, "target": target}

        for n_h in harmonics_list:
            for sf in stim_freqs:
                ref = create_reference(sf, fs, segment.shape[0], n_harmonics=n_h)
                feat[f"cca_{sf}Hz_h{n_h}"] = cca_score(segment, ref)

        features_list.append(feat)

    return pd.DataFrame(features_list)


def fbcca_scores(
    df: pd.DataFrame,
    eeg_cols: list[str],
    stim_freqs: list[int],
    n_subbands: int = 4,
    n_harmonics: int = 3,
    pre_sec: float = 0.5,
    win_sec: float = 6.85,
    fs: int = 256,
    band_high: float = 50.0,
) -> pd.DataFrame:
    """Filter-bank CCA (Chen et al., 2015), adapted to the 8–50 Hz preprocessed band.

    Sub-band b passband: [b * 8, band_high] Hz. Per-band weights w_b = b^(-1.25) + 0.25.
    Score(f) = sum_b w_b * r_b(f)^2, where r_b(f) is the CCA correlation between
    sub-band-filtered EEG and the reference at frequency f.
    """
    pre_samples = int(pre_sec * fs)
    win_samples = int(win_sec * fs)

    sos_banks = []
    for b in range(1, n_subbands + 1):
        lo = b * 8
        if lo >= band_high:
            break
        sos = signal.butter(4, [lo, band_high], btype="band", fs=fs, output="sos")
        sos_banks.append((b, sos))

    weights = np.array([b ** (-1.25) + 0.25 for b, _ in sos_banks])

    rows = []
    n_trials = int(df["trial"].max())
    for trial_num in range(1, n_trials + 1):
        trial_data = df[df["trial"] == trial_num]
        target = int(trial_data["stimulus"].iloc[0])
        eeg = trial_data[eeg_cols].to_numpy()

        segment = eeg[pre_samples : pre_samples + win_samples, :]
        if segment.shape[0] < win_samples:
            continue

        filtered = []
        for _, sos in sos_banks:
            filtered.append(signal.sosfiltfilt(sos, segment, axis=0))

        for sf in stim_freqs:
            ref = create_reference(sf, fs, segment.shape[0], n_harmonics=n_harmonics)
            r = np.array([cca_score(seg, ref) for seg in filtered])
            score = float(np.sum(weights * r**2))
            rows.append(
                {
                    "trial": trial_num,
                    "target": target,
                    "ref_hz": int(sf),
                    "score": score,
                }
            )

    return pd.DataFrame(rows)
