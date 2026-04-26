"""Filtering, re-referencing, and epoch extraction for SSVEP data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

try:
    from src.loader import Session, detect_trials, CH_NAMES
except ImportError:
    from loader import Session, detect_trials, CH_NAMES


@dataclass
class Epochs:
    data: np.ndarray  # (n_trials, n_channels, n_samples)
    labels: np.ndarray  # (n_trials,) — stimulation frequency in Hz
    sfreq: int
    tmin: float
    tmax: float
    ch_names: list[str]


def _butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    nyq = fs / 2
    return butter(order, [low / nyq, high / nyq], btype="band")


def _notch(freq: float, fs: int, q: float = 30.0):
    return iirnotch(freq, q, fs)


def filter_signal(eeg: np.ndarray, sfreq: int,
                  band: tuple[float, float] = (5.0, 45.0),
                  notch_hz: float | None = 50.0) -> np.ndarray:
    """Detrend, band-pass, optional notch, common-average reference."""
    out = eeg - eeg.mean(axis=1, keepdims=True)
    b, a = _butter_bandpass(band[0], band[1], sfreq)
    out = filtfilt(b, a, out, axis=1)
    if notch_hz is not None and notch_hz < sfreq / 2:
        bn, an = _notch(notch_hz, sfreq)
        out = filtfilt(bn, an, out, axis=1)
    car = out.mean(axis=0, keepdims=True)
    out = out - car
    return out


def epoch(session: Session, eeg_filt: np.ndarray,
          tmin: float = 0.0, tmax: float = 4.0) -> Epochs:
    onsets, _offsets, labels = detect_trials(session.trigger)
    fs = session.sfreq
    n_samp = int(round((tmax - tmin) * fs))
    n_total = eeg_filt.shape[1]
    starts = onsets + int(round(tmin * fs))
    keep = (starts >= 0) & (starts + n_samp <= n_total)
    starts, labels = starts[keep], labels[keep]

    data = np.empty((len(starts), eeg_filt.shape[0], n_samp), dtype=np.float64)
    for i, s in enumerate(starts):
        data[i] = eeg_filt[:, s : s + n_samp]

    return Epochs(data=data, labels=labels.astype(int), sfreq=fs,
                  tmin=tmin, tmax=tmax, ch_names=list(CH_NAMES))


def epochs_for_class(epochs: Epochs, freq: int) -> np.ndarray:
    return epochs.data[epochs.labels == freq]
