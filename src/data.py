from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from src.config import FEATURES_DIR, PROCESSED_DIR

_FNAME_RE = re.compile(r"subject_(\d+)_session_(\d+)\.csv$")


def list_sessions() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for p in PROCESSED_DIR.glob("subject_*_session_*.csv"):
        m = _FNAME_RE.search(p.name)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return sorted(pairs)


@lru_cache(maxsize=16)
def load_session(subject: int, session: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"subject_{subject}_session_{session}.csv"
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_psd_per_class() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DIR / "psd_per_class.csv")


@lru_cache(maxsize=1)
def load_snr_matrix() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DIR / "snr_matrix.csv")


@lru_cache(maxsize=1)
def load_cca_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DIR / "cca_features.csv")


@lru_cache(maxsize=1)
def load_fbcca_scores() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DIR / "fbcca_scores.csv")
