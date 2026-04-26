from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"

CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
TARGET_FREQS = [9, 10, 12, 15]
FS = 256

CLASS_COLORS: dict[int, str] = {
    9: "#1f77b4",
    10: "#2ca02c",
    12: "#ff7f0e",
    15: "#d62728",
}

PRE_SEC = 0.5
WIN_SEC = 6.85
CCA_HARMONICS = [1, 2, 3, 4, 5]
