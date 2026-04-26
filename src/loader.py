"""Load SSVEP recordings from g.tec .mat files into a tidy dict + MNE Raw."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
TARGET_FREQS = [9.0, 10.0, 12.0, 15.0]
LDA_TO_FREQ = {0: None, 1: 15.0, 2: 12.0, 3: 10.0, 4: 9.0}


@dataclass
class Session:
    subject: int
    session: int
    sfreq: int
    time: np.ndarray
    eeg: np.ndarray
    trigger: np.ndarray
    lda: np.ndarray
    raw: mne.io.RawArray

    @property
    def label(self) -> str:
        return f"subject_{self.subject}_session_{self.session}"

    @property
    def title(self) -> str:
        return f"Subject {self.subject} — Session {self.session}"


def _make_raw(eeg: np.ndarray, sfreq: int) -> mne.io.RawArray:
    info = mne.create_info(ch_names=list(CH_NAMES), sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg * 1e-6, info, verbose=False)  # MNE expects volts
    raw.set_montage("standard_1020", match_case=False, verbose=False)
    return raw


def load_session(path: str | Path, subject: int, session: int) -> Session:
    path = Path(path)
    m = loadmat(path)
    y = m["y"]
    sfreq = int(m["fs"].squeeze())
    if y.shape[0] != 11:
        raise ValueError(f"{path}: expected 11 channels, got {y.shape[0]}")

    time = y[0]
    eeg = y[1:9].astype(np.float64)
    trigger = y[9].astype(int)
    lda = y[10].astype(int)

    raw = _make_raw(eeg, sfreq)
    return Session(
        subject=subject, session=session, sfreq=sfreq,
        time=time, eeg=eeg, trigger=trigger, lda=lda, raw=raw,
    )


def discover_files(folder: str | Path) -> list[tuple[Path, int, int]]:
    folder = Path(folder)
    out: list[tuple[Path, int, int]] = []
    for subj in (1, 2):
        for sess in (1, 2):
            p = folder / "ssvep" / f"subject_{subj}_fvep_led_training_{sess}.mat"
            if p.exists():
                out.append((p, subj, sess))
    return out


def detect_trials(trigger: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    onsets = np.where((trigger[1:] != 0) & (trigger[:-1] == 0))[0] + 1
    offsets_all = np.where((trigger[1:] == 0) & (trigger[:-1] != 0))[0] + 1
    labels = trigger[onsets]
    offsets = np.empty_like(onsets)
    for i, on in enumerate(onsets):
        later = offsets_all[offsets_all > on]
        offsets[i] = later[0] if len(later) else len(trigger)
    return onsets, offsets, labels
