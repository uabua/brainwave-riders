"""Streamlit dashboard for SSVEP analysis.

Run with:
    streamlit run analysis/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import streamlit as st
import pandas as pd

from src.data import (
    list_sessions,
    load_cca_features,
    load_fbcca_scores,
    load_psd_per_class,
    load_session,
    load_snr_matrix,
)
from src.plots.cca import cca_matrix
from src.plots.fbcca import fbcca_matrix
from src.plots.psd import psd_per_class
from src.plots.raw import raw_overview
from src.plots.snr import snr_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / "data"


@st.cache_data
def _load(subject: int, session: int):
    return load_session(subject, session)


@st.cache_data
def _figure(subject: int, session: int):
    df = _load(subject, session)
    return raw_overview(df, title=f"Subject {subject} — Session {session}")


@st.cache_data
def _psd_df():
    return load_psd_per_class()


@st.cache_data
def _psd_figure(subject: int):
    return psd_per_class(_psd_df(), subject)


@st.cache_data
def _snr_df():
    return load_snr_matrix()


@st.cache_data
def _cca_df():
    return load_cca_features()


@st.cache_data
def _fbcca_df():
    return load_fbcca_scores()


@st.cache_data
def _snr_figure(subject: int):
    return snr_matrix(_snr_df(), subject)


@st.cache_data
def _cca_figure(subject: int, n_harmonics: int):
    return cca_matrix(_cca_df(), subject, n_harmonics=n_harmonics)


@st.cache_data
def _fbcca_figure(subject: int):
    return fbcca_matrix(_fbcca_df(), subject)


st.set_page_config(page_title="Brainwave Riders - SSVEP Dashboard", layout="wide")
st.title("SSVEP Analysis Dashboard")
st.caption(
    "Two subjects × two LED-flicker sessions, eight occipital electrodes "
    "(PO7, PO3, POz, PO4, PO8, O1, Oz, O2) at 256 Hz, 20 trials per session "
    "across 9 / 10 / 12 / 15 Hz targets."
)

tab_analysis, tab_ml = st.tabs(["Analysis", "ML Results"])

with tab_analysis:
    st.subheader("Dataset overview")
    overview = pd.DataFrame(
        [
            {
                "file": f"subject_{s}_fvep_led_training_{n}.mat",
                "subject": s,
                "session": n,
                "trials": 20,
            }
            for s, n in [(1, 1), (1, 2), (2, 1), (2, 2)]
        ]
    )
    st.dataframe(overview, width="stretch", hide_index=True)

    by_subject: dict[int, list[int]] = defaultdict(list)
    for subj, sess in list_sessions():
        by_subject[subj].append(sess)

    st.header("Raw signal overview")
    st.subheader(f"Subject {subj}")

    subjects = sorted(by_subject)
    for idx, subj in enumerate(subjects):

        if idx > 0:
            st.divider()

        st.subheader("Raw signal overview")
        sessions = sorted(by_subject[subj])
        cols = st.columns(len(sessions))
        for col, sess in zip(cols, sessions):
            with col:
                st.plotly_chart(_figure(subj, sess), use_container_width=True)

        st.subheader("PSD per class")
        st.plotly_chart(_psd_figure(subj), use_container_width=True)

        st.subheader("SNR matrix")
        st.caption(
            "Mean Welch SNR at the target frequency (h=1) across the "
            "post-onset window, per channel and class."
        )
        st.plotly_chart(_snr_figure(subj), use_container_width=True)

        st.subheader("CCA matrix")
        n_harm = st.select_slider(
            "Reference harmonics",
            options=[1, 2, 3, 4, 5],
            value=3,
            key=f"cca_h_{subj}",
        )
        st.caption(
            "Per-trial CCA correlation against each reference; trials "
            "sorted by true target so a strong block diagonal indicates "
            "good separability."
        )
        st.plotly_chart(_cca_figure(subj, n_harm), use_container_width=True)

        st.subheader("FBCCA matrix")
        st.caption(
            "Filter-bank CCA (4 sub-bands within 8–50 Hz, weighted "
            "sum of squared correlations)."
        )
        st.plotly_chart(_fbcca_figure(subj), use_container_width=True)
