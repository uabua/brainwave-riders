from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.config import CH_NAMES, FEATURES_DIR, TARGET_FREQS  # noqa: E402

PSD_PATH = FEATURES_DIR / "psd_features.csv"
OUT = FEATURES_DIR / "snr_matrix.csv"


def main() -> None:
    if not PSD_PATH.exists():
        raise SystemExit(
            f"Missing {PSD_PATH}. Run scripts/compute_features_psd.py first."
        )
    psd = pd.read_csv(PSD_PATH)

    rows = []
    for (subj, sess), grp in psd.groupby(["subject", "session"]):
        for target in TARGET_FREQS:
            cell = grp[grp["target"] == target]
            if cell.empty:
                continue
            for ci, ch_name in enumerate(CH_NAMES, start=1):
                col = f"ch{ci}_snr_{target}Hz_h1"
                if col not in cell.columns:
                    continue
                rows.append(
                    {
                        "subject": int(subj),
                        "session": int(sess),
                        "target_hz": int(target),
                        "channel": ch_name,
                        "snr": float(np.nanmean(cell[col].to_numpy())),
                    }
                )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
