"""Story-first Streamlit dashboard for the SSVEP dataset.

Run from the repo root:
    streamlit run webapp/app.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).resolve()
ROOT = HERE.parent
SRC_DIR = HERE / "src"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data" / "raw"
PICKLE_PATH = RESULTS_DIR / "data.pkl"
SIMULATOR_PATH = RESULTS_DIR / "simulator.html"

sys.path.insert(0, str(SRC_DIR))

from src.analysis import (  # noqa: E402
    accuracy_vs_window,
    cca_decode,
    cca_features,
    confusion as cca_confusion,
    online_lda_accuracy,
    project,
    psd_features,
    snr_at,
    snr_matrix,
    welch_psd,
)
from src.loader import CH_NAMES, LDA_TO_FREQ, Session, discover_files, load_session  # noqa: E402
import src.plots  # noqa: E402
import src.plots_plotly as pp  # noqa: E402
from src.preprocess import Epochs, epoch, filter_signal  # noqa: E402

WINDOWS = [1.0, 2.0, 3.0, 4.0]
TARGETS = [9.0, 10.0, 12.0, 15.0]
CLASSES = [9, 10, 12, 15]
OCCIPITAL_CHANS = ("O1", "Oz", "O2")

BG = "#f4efe6"
INK = "#181715"
MUTED = "#5f5a52"
PANEL = "#fffdf9"
EDGE = "#d7cdbd"
ACCENT = "#d95d39"
ACCENT_2 = "#3f6fd8"
ACCENT_3 = "#1f8a70"
ACCENT_4 = "#af7d2d"
SUB_COLORS = {1: ACCENT_2, 2: ACCENT}
METHOD_COLORS = {"PSD": "#8e8a82", "CCA": ACCENT_2, "FBCCA": ACCENT}


# ---------- cached compute layer ----------

@st.cache_data(show_spinner=False)
def get_available():
    return discover_files(DATA_DIR)


@st.cache_data(show_spinner=False)
def get_pickle_data() -> dict | None:
    if not PICKLE_PATH.exists():
        return None
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def has_umap() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource(show_spinner="Loading session…")
def get_session(subject: int, sess: int) -> Session:
    files = {(s, n): p for p, s, n in get_available()}
    path = files[(subject, sess)]
    return load_session(path, subject, sess)


@st.cache_data(show_spinner="Filtering EEG…")
def get_filtered_eeg(subject: int, sess: int) -> np.ndarray:
    session = get_session(subject, sess)
    return filter_signal(session.eeg, session.sfreq)


@st.cache_data(show_spinner="Building epochs…")
def get_epochs(subject: int, sess: int) -> Epochs:
    session = get_session(subject, sess)
    eeg_filt = get_filtered_eeg(subject, sess)
    return epoch(session, eeg_filt, tmin=0.0, tmax=4.0)


@st.cache_data(show_spinner="Computing SNR…")
def get_snr(subject: int, sess: int) -> np.ndarray:
    epochs = get_epochs(subject, sess)
    snr_mat, _ = snr_matrix(epochs, TARGETS)
    return snr_mat


@st.cache_data(show_spinner="Running CCA decoder…")
def get_cca(subject: int, sess: int) -> dict:
    epochs = get_epochs(subject, sess)
    preds, scores = cca_decode(epochs, TARGETS)
    cm = cca_confusion(epochs.labels, preds, CLASSES)
    acc_curve = accuracy_vs_window(epochs, TARGETS, WINDOWS)
    return {"preds": preds, "scores": scores, "cm": cm, "acc_curve": acc_curve}


@st.cache_data(show_spinner="Building PSD features…")
def get_psd_features(subject: int, sess: int) -> pd.DataFrame:
    epochs = get_epochs(subject, sess)
    occip_idx = [CH_NAMES.index(c) for c in OCCIPITAL_CHANS]
    return psd_features(epochs, occip_idx, ch_names=list(CH_NAMES))


@st.cache_data(show_spinner="Building CCA features…")
def get_cca_features(subject: int, sess: int) -> pd.DataFrame:
    epochs = get_epochs(subject, sess)
    cca = get_cca(subject, sess)
    return cca_features(cca["scores"], epochs.labels, TARGETS)


def _features_for(kind: str, subject: int, sess: int) -> pd.DataFrame:
    return get_psd_features(subject, sess) if kind == "psd" else get_cca_features(subject, sess)


@st.cache_data(show_spinner="Projecting session…")
def get_session_projection(subject: int, sess: int, kind: str, method: str) -> dict:
    df = _features_for(kind, subject, sess)
    coords, axis_labels = project(df, method)
    return {"coords": coords, "axis_labels": axis_labels, "targets": df["target"].to_numpy()}


@st.cache_data(show_spinner="Projecting subject…")
def get_subject_projection(subject: int, kind: str, method: str) -> dict:
    sess_list = sorted({n for _, s, n in get_available() if s == subject})
    parts = []
    for sess_n in sess_list:
        df = _features_for(kind, subject, sess_n).copy()
        df["session"] = sess_n
        parts.append(df)
    combined = pd.concat(parts, ignore_index=True)
    coords, axis_labels = project(combined, method)
    return {
        "coords": coords,
        "axis_labels": axis_labels,
        "targets": combined["target"].to_numpy(),
        "sessions": combined["session"].to_numpy(),
    }


@st.cache_data(show_spinner="Computing online LDA accuracy…")
def get_lda_acc(subject: int, sess: int) -> float:
    session = get_session(subject, sess)
    acc, _, _ = online_lda_accuracy(session, LDA_TO_FREQ)
    return acc


@st.cache_data(show_spinner="Aggregating cross-session results…")
def get_cross_results() -> dict:
    occip_idx = [CH_NAMES.index(c) for c in OCCIPITAL_CHANS]
    snr_records, acc_rows, cms, summary_rows = [], [], [], []

    for _path, subj, sess in get_available():
        epochs = get_epochs(subj, sess)
        snr_mat = get_snr(subj, sess)
        cca = get_cca(subj, sess)
        lda_acc = get_lda_acc(subj, sess)
        session = get_session(subj, sess)

        for class_idx, target in enumerate(TARGETS):
            snr_records.append(
                {
                    "subject": subj,
                    "session": sess,
                    "freq": int(target),
                    "snr": float(np.nanmean(snr_mat[class_idx, occip_idx])),
                }
            )

        acc_rows.append({"label": session.title, "windows": cca["acc_curve"], "lda_acc": lda_acc})
        cms.append({"label": session.title, "cm": cca["cm"], "classes": CLASSES})

        freqs, psd = welch_psd(epochs)
        for target in TARGETS:
            sel = epochs.labels == int(target)
            n_trials = int(sel.sum())
            if n_trials == 0:
                mean_snr = float("nan")
                cls_acc = float("nan")
            else:
                snr = snr_at(freqs, psd[sel][:, occip_idx, :], target)
                mean_snr = float(np.nanmean(snr))
                cls_acc = float((cca["preds"][sel] == int(target)).mean())
            summary_rows.append(
                {
                    "subject": subj,
                    "session": sess,
                    "class_hz": int(target),
                    "n_trials": n_trials,
                    "occipital_snr": mean_snr,
                    "cca_accuracy_4s": cls_acc,
                    "online_lda_accuracy": lda_acc,
                }
            )

    return {
        "snr_records": snr_records,
        "acc_rows": acc_rows,
        "cms": cms,
        "summary": pd.DataFrame(summary_rows),
    }


# ---------- UI helpers ----------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, #fff7ee 0, transparent 34%),
                radial-gradient(circle at top right, #edf2ff 0, transparent 28%),
                linear-gradient(180deg, #f7f1e6 0%, #f1eadf 100%);
            color: {INK};
        }}
        .block-container {{
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3 {{
            color: {INK};
        }}
        .hero {{
            background: linear-gradient(135deg, rgba(255,253,249,0.97), rgba(245,237,223,0.97));
            border: 1px solid {EDGE};
            border-radius: 28px;
            padding: 1.7rem 1.8rem 1.5rem 1.8rem;
            box-shadow: 0 20px 48px rgba(84, 63, 28, 0.08);
            margin-bottom: 1rem;
        }}
        .eyebrow {{
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            font-weight: 800;
            color: {ACCENT};
            margin-bottom: 0.55rem;
        }}
        .hero-title {{
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            font-size: 3.1rem;
            line-height: 0.95;
            margin: 0;
        }}
        .hero-copy {{
            color: {MUTED};
            max-width: 58rem;
            font-size: 1rem;
            line-height: 1.5;
            margin-top: 0.75rem;
        }}
        .metric-card {{
            background: {PANEL};
            border: 1px solid {EDGE};
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 144px;
            box-shadow: 0 10px 24px rgba(84, 63, 28, 0.05);
        }}
        .metric-label {{
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {MUTED};
            font-weight: 800;
        }}
        .metric-value {{
            font-size: 2rem;
            line-height: 1;
            font-weight: 800;
            color: {INK};
            margin: 0.45rem 0;
        }}
        .metric-note {{
            font-size: 0.92rem;
            line-height: 1.35;
            color: {MUTED};
        }}
        .story-card {{
            background: rgba(255,255,255,0.72);
            border: 1px solid {EDGE};
            border-radius: 24px;
            padding: 1rem 1.05rem 1.05rem 1.05rem;
            box-shadow: 0 10px 24px rgba(84, 63, 28, 0.04);
            margin-bottom: 1rem;
        }}
        .story-title {{
            font-size: 1.18rem;
            font-weight: 800;
            color: {INK};
            margin-bottom: 0.25rem;
        }}
        .story-copy {{
            color: {MUTED};
            line-height: 1.45;
        }}
        .quote-card {{
            background: linear-gradient(135deg, #1f1c18 0%, #3a3228 100%);
            color: #fff8ef;
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .quote-card strong {{
            color: #ffd8b6;
        }}
        .micro {{
            color: {MUTED};
            font-size: 0.83rem;
        }}
        div[data-testid="stTabs"] button {{
            font-weight: 700;
        }}
        .sim-wrap {{
            background: #0e1117;
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 24px 56px rgba(30,20,10,0.18);
            margin: 0.5rem 0 1rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-note'>{note}</div>"
        "</div>"
    )


def story_block(title: str, body: str) -> None:
    st.markdown(
        "<div class='story-card'>"
        f"<div class='story-title'>{title}</div>"
        f"<div class='story-copy'>{body}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def quote_block(title: str, body: str) -> None:
    st.markdown(
        f"<div class='quote-card'><strong>{title}</strong><br>{body}</div>",
        unsafe_allow_html=True,
    )


def pct(value: float) -> str:
    return f"{value:.0%}"


def bits(value: float) -> str:
    return f"{value:.1f} b/m"


def subject_session_options():
    by_subject: dict[int, list[int]] = {}
    for _path, subj, sess in get_available():
        by_subject.setdefault(subj, []).append(sess)
    for subj in by_subject:
        by_subject[subj] = sorted(by_subject[subj])
    return by_subject


def get_story_fallback(cross: dict) -> dict:
    summary_df = cross["summary"]
    sub_stats = (
        summary_df.groupby("subject")
        .agg(
            mean_occipital_snr=("occipital_snr", "mean"),
            mean_cca_accuracy=("cca_accuracy_4s", "mean"),
            mean_lda_accuracy=("online_lda_accuracy", "mean"),
        )
        .reset_index()
    )
    return {"cross_summary": sub_stats}


def fig_method_story(data: dict) -> go.Figure:
    pa = data["pipeline_avg"]
    svm = pa[pa["classifier"] == "SVM"].copy()
    methods = ["PSD", "CCA", "FBCCA"]
    svm = svm[svm["feat"].isin(methods)]
    svm["feat"] = pd.Categorical(svm["feat"], categories=methods, ordered=True)
    svm = svm.sort_values("feat")

    fig = go.Figure()
    for subj in [1, 2]:
        sub = svm[svm["subject"] == subj]
        fig.add_trace(
            go.Bar(
                name=f"Subject {subj}",
                x=sub["feat"],
                y=sub["accuracy_mean"],
                marker_color=SUB_COLORS[subj],
                error_y=dict(type="data", array=sub["accuracy_std"].tolist(), visible=True),
                text=[pct(v) for v in sub["accuracy_mean"]],
                textposition="outside",
            )
        )

    fig.add_hline(y=0.25, line_dash="dash", line_color=MUTED)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        title="Method progression: PSD → CCA → FBCCA",
        barmode="group",
        yaxis=dict(title="Mean LOSO Accuracy", range=[0, 1.12], tickformat=".0%"),
        xaxis=dict(title="Feature method"),
        height=430,
        margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", y=1.05, x=0.65),
    )
    return fig


def fig_channel_story(data: dict) -> go.Figure:
    pa = data["pipeline_avg"]
    svm = pa[pa["classifier"] == "SVM"].copy()
    label_map = {
        "CCA": ("CCA", "8-channel"),
        "CCA-3ch": ("CCA", "3-channel"),
        "FBCCA": ("FBCCA", "8-channel"),
        "FBCCA-3ch": ("FBCCA", "3-channel"),
    }
    df = svm[svm["feat"].isin(label_map)].copy()
    df["method"] = df["feat"].map(lambda x: label_map[x][0])
    df["channels"] = df["feat"].map(lambda x: label_map[x][1])

    fig = go.Figure()
    colors = {"8-channel": ACCENT_2, "3-channel": ACCENT_4}
    for channel_label in ["8-channel", "3-channel"]:
        sub = df[df["channels"] == channel_label]
        x = [f"{row.method}<br>Subject {row.subject}" for row in sub.itertuples()]
        fig.add_trace(
            go.Bar(
                name=channel_label,
                x=x,
                y=sub["accuracy_mean"],
                marker_color=colors[channel_label],
                text=[pct(v) for v in sub["accuracy_mean"]],
                textposition="outside",
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        title="Channel coverage matters only for the hard subject",
        barmode="group",
        yaxis=dict(title="Mean LOSO Accuracy", range=[0, 1.12], tickformat=".0%"),
        height=430,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def fig_fbcca_window_story(data: dict) -> go.Figure:
    sw = data["sliding_window_avg"]
    itr = data["itr"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=["Subject 1", "Subject 2"],
        horizontal_spacing=0.1,
    )

    for col, subj in enumerate([1, 2], 1):
        acc = sw[(sw["method"] == "FBCCA") & (sw["subject"] == subj)].sort_values("win_sec")
        itr_sub = itr[(itr["method"] == "FBCCA") & (itr["subject"] == subj)].sort_values("win_sec")
        peak = itr_sub.loc[itr_sub["itr"].idxmax()]

        fig.add_trace(
            go.Scatter(
                x=acc["win_sec"],
                y=acc["accuracy_mean"],
                mode="lines+markers",
                name=f"Accuracy · Subject {subj}",
                line=dict(color=SUB_COLORS[subj], width=3),
                marker=dict(size=8),
                legendgroup=f"acc{subj}",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=itr_sub["win_sec"],
                y=itr_sub["itr"],
                mode="lines+markers",
                name="ITR",
                line=dict(color=ACCENT_3, width=2, dash="dot"),
                marker=dict(size=7, color=ACCENT_3),
                legendgroup="itr",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
            secondary_y=True,
        )
        fig.add_vline(x=peak["win_sec"], line_dash="dot", line_color=ACCENT, row=1, col=col)
        fig.add_annotation(
            x=peak["win_sec"],
            y=peak["itr"],
            text=f"{peak['itr']:.1f} b/m",
            showarrow=True,
            arrowhead=2,
            arrowcolor=ACCENT,
            ax=18,
            ay=-28,
            font=dict(color=ACCENT, size=10),
            row=1,
            col=col,
        )

    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0.6, 1.05], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="",          tickformat=".0%", range=[0.6, 1.05], row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="ITR (bits/min)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="",              row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="Window length (s)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        title="FBCCA window tradeoff: speed and certainty do not peak together",
        height=430,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def render_hero(pickle_data: dict | None) -> None:
    if pickle_data is not None and "story" in pickle_data:
        story = pickle_data["story"]
        summary = pickle_data["summary"]
        subtitle = story["callouts"]["headline"]
        metric_values = [
            ("Best Accuracy", pct(summary["fbcca_acc_sub1"]), "FBCCA + SVM reaches 100% on both subjects."),
            ("Largest Rescue", f"{story['baseline']['fbcca_gain_vs_psd_sub2'] * 100:.0f} pts", "Subject 2 jumps from PSD failure to FBCCA reliability."),
            ("Peak ITR", bits(summary["best_itr_sub1"]), "Subject 1 peaks at 1 second, before perfect accuracy."),
            ("Adaptive Window", f"{story['timing']['fbcca_95pct_window_sub1']:.0f}s / {story['timing']['fbcca_95pct_window_sub2']:.0f}s", "95% of max accuracy arrives fast for Subject 1 and late for Subject 2."),
            ("Dataset Scope", "80 trials", "2 subjects × 2 sessions × 4 stimulation classes."),
        ]
    else:
        subtitle = (
            "This interface walks from raw evoked evidence to decoder geometry and finally "
            "to the deployment tradeoff: fast enough for one subject is not enough for another."
        )
        metric_values = [
            ("Signals", "8 EEG", "PO7, PO3, POz, PO4, PO8, O1, Oz, O2"),
            ("Sampling", "256 Hz", "Consistent across all recordings."),
            ("Trials", "80", "20 trials per session."),
            ("Classes", "4", "9 / 10 / 12 / 15 Hz"),
            ("Sessions", "4", "2 subjects × 2 runs."),
        ]

    st.markdown(
        "<div class='hero'>"
        "<div class='eyebrow'>Brainwave Riders · Story Dashboard</div>"
        "<div class='hero-title'>One dashboard, two very different brains.</div>"
        f"<div class='hero-copy'>{subtitle}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    for col, (label, value, note) in zip(cols, metric_values):
        with col:
            st.markdown(metric_card(label, value, note), unsafe_allow_html=True)


def render_signal_tab(pickle_data: dict | None) -> None:
    story_text = (
        pickle_data["story"]["callouts"]["tab1"]
        if pickle_data is not None and "story" in pickle_data
        else "Start with the session-level evidence: SNR, PSD shape, and time-frequency structure."
    )
    story_block(
        "The BCI-illiteracy spectrum",
        "This section anchors the whole narrative in the raw physiology. We look at one session at a time, "
        "starting from the hard responder by default, because that is where weak SSVEP structure becomes visible.",
    )
    quote_block("Signal takeaway", story_text)

    subjects = subject_session_options()
    control1, control2, control3 = st.columns([1, 1, 2])
    with control1:
        subject = st.radio("Subject", sorted(subjects), horizontal=True, index=1, key="signal_subject")
    with control2:
        sessions = subjects[subject]
        default_idx = 0 if len(sessions) == 1 else min(1, len(sessions) - 1)
        sess = st.radio("Session", sessions, horizontal=True, index=default_idx, key="signal_session")
    with control3:
        st.markdown(
            "<div class='micro'>The layout stays one or two charts wide so it remains readable on smaller laptop screens.</div>",
            unsafe_allow_html=True,
        )

    session = get_session(subject, sess)
    eeg_filt = get_filtered_eeg(subject, sess)
    epochs = get_epochs(subject, sess)
    snr_mat = get_snr(subject, sess)
    lda_acc = get_lda_acc(subject, sess)

    upper_left, upper_right = st.columns(2)
    with upper_left:
        st.plotly_chart(pp.raw_overview(session, eeg_filt), use_container_width=True, key=f"raw_{subject}_{sess}")
    with upper_right:
        st.plotly_chart(pp.rest_vs_stimulus(session, eeg_filt), use_container_width=True, key=f"reststim_{subject}_{sess}")

    mid_left, mid_right = st.columns(2)
    with mid_left:
        st.plotly_chart(pp.psd_per_class(epochs, session), use_container_width=True, key=f"psdcls_{subject}_{sess}")
    with mid_right:
        st.plotly_chart(pp.snr_heatmap(snr_mat, session), use_container_width=True, key=f"snrhm_{subject}_{sess}")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        occip_idx = [CH_NAMES.index(c) for c in OCCIPITAL_CHANS]
        st.plotly_chart(
            pp.snr_class_heatmap(epochs, session, ch_indices=occip_idx),
            use_container_width=True,
            key=f"snrclass_{subject}_{sess}",
        )
    with lower_right:
        st.plotly_chart(pp.time_frequency(session, eeg_filt), use_container_width=True, key=f"tf_{subject}_{sess}")

    facts = st.columns(4)
    mean_occipital_snr = float(np.nanmean(snr_mat[:, [CH_NAMES.index(c) for c in OCCIPITAL_CHANS]]))
    facts[0].metric("Online LDA", pct(lda_acc))
    facts[1].metric("Mean occipital SNR", f"{mean_occipital_snr:.2f}")
    facts[2].metric("Trials in session", str(len(epochs.labels)))
    facts[3].metric("Class balance", "5 each")


def render_methods_tab(pickle_data: dict | None) -> None:
    story_text = (
        pickle_data["story"]["callouts"]["tab2"]
        if pickle_data is not None and "story" in pickle_data
        else "Move from physiology into representation: which feature space actually separates the classes?"
    )
    story_block(
        "Comparative methodology",
        "Now we move from visible signal quality to the internal shape of the decoder evidence. "
        "The key question is not only who wins, but why the classes become separable.",
    )
    quote_block("Method takeaway", story_text)

    if pickle_data is not None:
        row1_left, row1_right = st.columns([1.1, 1])
        with row1_left:
            st.plotly_chart(fig_method_story(pickle_data), use_container_width=True)
        with row1_right:
            st.plotly_chart(fig_channel_story(pickle_data), use_container_width=True)

    subjects = subject_session_options()
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        subject = st.radio("Feature subject", sorted(subjects), horizontal=True, index=1, key="method_subject")
    with c2:
        method = st.radio("Projection", ["pca", "umap"] if has_umap() else ["pca"], horizontal=True, key="proj_method")
    with c3:
        sess = st.radio("Evidence session", subjects[subject], horizontal=True, key="method_session")

    proj_cca = get_subject_projection(subject, "cca", method)
    proj_psd = get_subject_projection(subject, "psd", method)
    session = get_session(subject, sess)
    epochs = get_epochs(subject, sess)
    cca = get_cca(subject, sess)

    proj_left, proj_right = st.columns(2)
    with proj_left:
        fig = pp.feature_space_3d_overlay(
            proj_cca["coords"],
            proj_cca["axis_labels"],
            proj_cca["targets"],
            proj_cca["sessions"],
            title=f"CCA feature space · Subject {subject}",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"cca_overlay_{subject}_{method}")
    with proj_right:
        fig = pp.feature_space_3d_overlay(
            proj_psd["coords"],
            proj_psd["axis_labels"],
            proj_psd["targets"],
            proj_psd["sessions"],
            title=f"PSD feature space · Subject {subject}",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"psd_overlay_{subject}_{method}")

    heat_left, heat_right = st.columns(2)
    with heat_left:
        st.plotly_chart(
            pp.cca_correlation_heatmap(cca["scores"], epochs.labels, session),
            use_container_width=True,
            key=f"cca_corr_{subject}_{sess}",
        )
    with heat_right:
        st.plotly_chart(
            pp.confusion(cca["cm"], CLASSES, session, label="CCA"),
            use_container_width=True,
            key=f"conf_{subject}_{sess}",
        )

    with st.expander("Show class counts for this session"):
        st.plotly_chart(pp.class_distribution(epochs, session), use_container_width=True, key=f"classdist_{subject}_{sess}")


def render_utility_tab(pickle_data: dict | None) -> None:
    story_text = (
        pickle_data["story"]["callouts"]["tab3"]
        if pickle_data is not None and "story" in pickle_data
        else "The deployment question is not only which method is accurate, but which one stays useful across time budgets."
    )
    story_block(
        "The neuroscience of utility",
        "This is the deployment layer. A real interface has to translate signal quality into usable throughput, "
        "which is why window length, online classifier quality, and cross-session consistency all matter.",
    )
    quote_block("Utility takeaway", story_text)

    cross = get_cross_results()

    top_left, top_right = st.columns([1.05, 1])
    with top_left:
        if pickle_data is not None:
            st.plotly_chart(fig_fbcca_window_story(pickle_data), use_container_width=True)
        else:
            st.plotly_chart(pp.accuracy_curves(cross["acc_rows"]), use_container_width=True, key="acc_curves_fallback")
    with top_right:
        st.plotly_chart(pp.accuracy_curves(cross["acc_rows"]), use_container_width=True, key="acc_curves")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(pp.snr_comparison(cross["snr_records"]), use_container_width=True, key="snr_comp")
    with bottom_right:
        st.plotly_chart(pp.confusion_grid(cross["cms"]), use_container_width=True, key="conf_grid")

    with st.expander("Cross-session summary table"):
        summary_df = cross["summary"].copy()
        summary_df["occipital_snr"] = summary_df["occipital_snr"].round(2)
        summary_df["cca_accuracy_4s"] = summary_df["cca_accuracy_4s"].map(pct)
        summary_df["online_lda_accuracy"] = summary_df["online_lda_accuracy"].map(pct)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_simulator_tab() -> None:
    story_block(
        "Interactive window-length simulator",
        "Use the slider to step through decision windows from 1 s to 6.85 s. "
        "Watch how FBCCA confidence (accuracy %) and information throughput (ITR) evolve "
        "as more data cycles through the filter bank — and where Subject 1 and Subject 2 peak.",
    )
    if SIMULATOR_PATH.exists():
        st.markdown("<div class='sim-wrap'>", unsafe_allow_html=True)
        html = SIMULATOR_PATH.read_text(encoding="utf-8")
        components.html(html, height=640, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(
            f"Simulator not found at `{SIMULATOR_PATH}`. "
            "Run `python build_simulator.py` from the repo root to generate it."
        )


def render_explorer_tab() -> None:
    story_block(
        "Session explorer",
        "This is the deep-dive zone. The core story stays curated above, but you can still pull any session apart here without crowding the main narrative.",
    )

    subjects = subject_session_options()
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        subject = st.selectbox("Subject", sorted(subjects), index=0, key="explore_subject")
    with c2:
        sess = st.selectbox("Session", subjects[subject], index=0, key="explore_session")
    with c3:
        chosen = st.multiselect(
            "Plots to show",
            [
                "Raw overview",
                "Rest vs stimulus",
                "PSD per channel",
                "Topomap",
                "Time-frequency",
                "CCA confusion",
            ],
            default=["Raw overview", "PSD per channel", "CCA confusion"],
            key="explore_plots",
        )

    session = get_session(subject, sess)
    eeg_filt = get_filtered_eeg(subject, sess)
    epochs = get_epochs(subject, sess)
    cca = get_cca(subject, sess)

    if "Raw overview" in chosen:
        st.plotly_chart(pp.raw_overview(session, eeg_filt), use_container_width=True, key=f"explore_raw_{subject}_{sess}")
    if "Rest vs stimulus" in chosen:
        st.plotly_chart(pp.rest_vs_stimulus(session, eeg_filt), use_container_width=True, key=f"explore_rvs_{subject}_{sess}")
    if "PSD per channel" in chosen:
        st.plotly_chart(pp.psd_per_channel(epochs, session), use_container_width=True, key=f"explore_psdch_{subject}_{sess}")
    if "Topomap" in chosen:
        fig_topo = plots.plot_topomaps(epochs, session, out=RESULTS_DIR / "figures", save=False)
        st.pyplot(fig_topo, use_container_width=True)
    if "Time-frequency" in chosen:
        st.plotly_chart(pp.time_frequency(session, eeg_filt), use_container_width=True, key=f"explore_tf_{subject}_{sess}")
    if "CCA confusion" in chosen:
        st.plotly_chart(pp.confusion(cca["cm"], CLASSES, session, label="CCA"), use_container_width=True, key=f"explore_conf_{subject}_{sess}")


def main() -> None:
    st.set_page_config(
        page_title="Brainwave Riders · SSVEP Story Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    available = get_available()
    if not available:
        st.error(f"No `subject_*_fvep_led_training_*.mat` files found under {DATA_DIR / 'ssvep'}")
        st.stop()

    pickle_data = get_pickle_data()

    render_hero(pickle_data)
    story_block(
        "Dataset at a glance",
        "Two subjects, two sessions each, eight posterior electrodes, 256 Hz sampling, and 20 trials per recording across 9 / 10 / 12 / 15 Hz targets. "
        "The story below moves from raw evidence, to separability, to deployment tradeoffs.",
    )

    tabs = st.tabs([
        "1. Signal Fragility",
        "2. Methods",
        "3. Utility",
        "4. Simulator",
        "5. Explorer",
    ])

    with tabs[0]:
        render_signal_tab(pickle_data)
    with tabs[1]:
        render_methods_tab(pickle_data)
    with tabs[2]:
        render_utility_tab(pickle_data)
    with tabs[3]:
        render_simulator_tab()
    with tabs[4]:
        render_explorer_tab()


if __name__ == "__main__":
    main()
