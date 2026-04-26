from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.config import CH_NAMES, FEATURES_DIR, FS, TARGET_FREQS  # noqa: E402
from src.data import list_sessions, load_session  # noqa: E402

NPERSEG = 1024
NOVERLAP = 768

OUT = FEATURES_DIR / "psd_per_class.csv"


def main() -> None:
    rows: list[tuple] = []
    for subj, sess in list_sessions():
        df = load_session(subj, sess)
        for target in TARGET_FREQS:
            trial = df.loc[df["stimulus"] == target, CH_NAMES].to_numpy()
            if trial.shape[0] < NPERSEG:
                continue
            psds = []
            freqs = None
            for ci in range(len(CH_NAMES)):
                f, p = signal.welch(
                    trial[:, ci], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP
                )
                psds.append(p)
                freqs = f
            mean_psd = np.mean(psds, axis=0)
            for fi, pi in zip(freqs, mean_psd):
                rows.append((subj, sess, int(target), float(fi), float(pi)))

    out = pd.DataFrame(rows, columns=["subject", "session", "target_hz", "freq", "psd"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
