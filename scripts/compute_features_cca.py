from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.config import (
    CCA_HARMONICS,
    CH_NAMES,
    FEATURES_DIR,
    FS,
    PRE_SEC,  # noqa: E402
    TARGET_FREQS,
    WIN_SEC,
)
from src.data import list_sessions, load_session  # noqa: E402

from _features_common import extract_features_cca  # noqa: E402

OUT = FEATURES_DIR / "cca_features.csv"


def main() -> None:
    parts = []
    for subj, sess in list_sessions():
        df = load_session(subj, sess)
        feats = extract_features_cca(
            df,
            CH_NAMES,
            TARGET_FREQS,
            CCA_HARMONICS,
            pre_sec=PRE_SEC,
            win_sec=WIN_SEC,
            fs=FS,
        )
        feats.insert(0, "session", sess)
        feats.insert(0, "subject", subj)
        parts.append(feats)
    out = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows × {len(out.columns)} cols -> {OUT}")


if __name__ == "__main__":
    main()
