from __future__ import annotations

from pathlib import Path

import scipy.io as sio
import pandas as pd
import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent

DATA_DIR = HERE.parent / "data"

CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
TARGET_FREQS = [9.0, 10.0, 12.0, 15.0]
FS = 256


def load(filepath: str) -> pd.DataFrame:
    data = sio.loadmat(filepath)["y"].T

    df = pd.DataFrame(
        data,
        columns=["timestamp"]
        + CH_NAMES
        + [
            "stimulus",
            "classifier_output",
        ],
    )

    # Extract trial onsets and assign labels
    trigger = df["stimulus"].to_numpy()
    trigger_on = trigger > 0
    onsets = np.where(np.diff(trigger_on.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(trigger_on.astype(int)) == -1)[0] + 1

    df["trial"] = 0
    for trial_n, (on, off) in enumerate(zip(onsets, offsets)):
        df.loc[on : off - 1, "trial"] = trial_n + 1

    df["stimulus"] = df["stimulus"].astype(int)
    df["classifier_output"] = df["classifier_output"].astype(int)

    return df


def preprocess(df: pd.DataFrame, eeg_cols: list[str], fs: int = 256) -> pd.DataFrame:
    df = df.copy()

    # Keep 8-50 Hz, remove everything else
    sos = signal.butter(4, [8, 50], btype="band", fs=fs, output="sos")
    # Apply filter to each EEG channel
    for col in eeg_cols:
        df[col] = signal.sosfiltfilt(sos, df[col].to_numpy())

    return df


if __name__ == "__main__":
    sub_1_sess_1 = preprocess(
        load(
            f"{DATA_DIR}/raw/subject_1_fvep_led_training_1.mat",
        ),
        eeg_cols=CH_NAMES,
        fs=FS,
    )
    sub_1_sess_1.to_csv(f"{DATA_DIR}/processed/subject_1_session_1.csv", index=False)

    sub_1_sess_2 = preprocess(
        load(
            f"{DATA_DIR}/raw/subject_1_fvep_led_training_2.mat",
        ),
        eeg_cols=CH_NAMES,
        fs=FS,
    )
    sub_1_sess_2.to_csv(f"{DATA_DIR}/processed/subject_1_session_2.csv", index=False)

    sub_2_sess_1 = preprocess(
        load(
            f"{DATA_DIR}/raw/subject_2_fvep_led_training_1.mat",
        ),
        eeg_cols=CH_NAMES,
        fs=FS,
    )
    sub_2_sess_1.to_csv(f"{DATA_DIR}/processed/subject_2_session_1.csv", index=False)

    sub_2_sess_2 = preprocess(
        load(
            f"{DATA_DIR}/raw/subject_2_fvep_led_training_2.mat",
        ),
        eeg_cols=CH_NAMES,
        fs=FS,
    )
    sub_2_sess_2.to_csv(f"{DATA_DIR}/processed/subject_2_session_2.csv", index=False)
