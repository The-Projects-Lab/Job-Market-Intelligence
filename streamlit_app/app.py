
import os
import sys
import re
import glob
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# ============================================================
# PATH & ENVIRONMENT SETUP
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

# Ensure project root is in Python module search path
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Force load .env from project root
load_dotenv(BASE_DIR / ".env")

GOLD_DIR = BASE_DIR / "data" / "gold_v3"
MODEL_DIR = BASE_DIR / "models"

JOBS_PATH = GOLD_DIR / "jobs_gold.csv.gz"
SKILL_DEMAND_PATH = GOLD_DIR / "skill_demand.csv.gz"
ROLE_DEMAND_PATH = GOLD_DIR / "role_demand.csv.gz"
COMPANY_DEMAND_PATH = GOLD_DIR / "company_demand.csv.gz"
LOCATION_DEMAND_PATH = GOLD_DIR / "location_demand.csv.gz"
SALARY_INSIGHTS_PATH = GOLD_DIR / "salary_insights.csv.gz"
SOURCE_QUALITY_PATH = GOLD_DIR / "source_quality.csv.gz"

RECOMMENDER_DIR = MODEL_DIR / "job_recommender"
RECOMMENDER_VECTORIZER_PATH = RECOMMENDER_DIR / "tfidf_vectorizer.pkl"
RECOMMENDER_VECTORS_PATH = RECOMMENDER_DIR / "job_vectors.pkl"
RECOMMENDER_JOBS_PATH = RECOMMENDER_DIR / "recommender_jobs.csv"

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None


# ============================================================
# BASIC CONFIG
# ============================================================

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM — "SIGNAL" TOKENS
# Two-tone gradient (signal blue → violet) is the one signature
# accent used across the app: the GenAI brand, KPI card edges, and
# the chat avatar. Mint = available/positive, amber = gap/missing —
# consistent semantics everywhere they appear.
# ============================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-base: #0f111a;
        --bg-panel: #171a26;
        --bg-panel-raised: #1e2233;
        --signal-blue: #4c7cf6;
        --signal-violet: #9d6bff;
        --signal-gradient: linear-gradient(135deg, var(--signal-blue) 0%, var(--signal-violet) 100%);
        --mint: #35d0a0;
        --mint-dim: rgba(53, 208, 160, 0.14);
        --amber: #f5a623;
        --amber-dim: rgba(245, 166, 35, 0.14);
        --text-primary: #eef1f7;
        --text-muted: #8d94a8;
        --border-subtle: rgba(238, 241, 247, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-base);
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 96%;
    }

    h1, h2, h3 { font-family: 'Sora', sans-serif; letter-spacing: -0.01em; }

    /* ---------- Hero ---------- */
    .hero-band {
        background: linear-gradient(135deg, #12141f 0%, #171a2c 55%, #1c1a33 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 24px;
    }
    .hero-eyebrow {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: var(--signal-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        color: var(--text-primary);
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 6px;
        max-width: 720px;
    }

    /* ---------- KPI cards — gradient signal edge ---------- */
    .kpi-card {
        position: relative;
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px 20px 16px 22px;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: var(--signal-gradient);
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 26px -10px rgba(0, 0, 0, 0.55);
        border-color: rgba(157, 107, 255, 0.35);
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'Sora', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .kpi-sub { font-size: 0.76rem; color: var(--text-muted); margin-top: 6px; }

    /* ---------- Section labels ---------- */
    .section-label {
        font-family: 'Sora', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 26px 0 12px 0;
        padding-left: 12px;
        border-left: 3px solid var(--signal-violet);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #0c0e16;
        border-right: 1px solid var(--border-subtle);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 18px 0;
    }
    .sidebar-orb {
        width: 30px; height: 30px; border-radius: 50%;
        background: var(--signal-gradient);
        box-shadow: 0 0 14px rgba(157, 107, 255, 0.55);
    }
    .sidebar-brand-text {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.98rem;
        color: var(--text-primary);
        line-height: 1.15;
    }
    .sidebar-brand-sub { font-size: 0.7rem; color: var(--text-muted); }

    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {
        background: transparent;
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 2px;
        transition: background 0.18s ease;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {
        background: rgba(157, 107, 255, 0.08);
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border-subtle);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: var(--signal-violet);
    }
    div[data-testid="stFormSubmitButton"] button,
    .stButton > button[kind="primary"] {
        background: var(--signal-gradient) !important;
        color: white !important;
        border: none !important;
    }

    /* ---------- Tabs / plotly / dataframe ---------- */
    .stPlotlyChart {
        border-radius: 14px;
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        padding: 6px;
        transition: transform 0.22s ease, border-color 0.22s ease;
    }
    .stPlotlyChart:hover { transform: translateY(-2px); border-color: rgba(76, 124, 246, 0.35); }
    .stDataFrame { font-family: 'JetBrains Mono', monospace; }

    /* ---------- Skill chips ---------- */
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid var(--border-subtle);
        transition: transform 0.15s ease;
    }
    .chip:hover { transform: translateY(-2px); }
    .chip-available { background: var(--mint-dim); color: var(--mint); border-color: rgba(53,208,160,0.35); }
    .chip-missing { background: var(--amber-dim); color: var(--amber); border-color: rgba(245,166,35,0.35); }

    /* ---------- Job recommendation cards ---------- */
    .job-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .job-card:hover { transform: translateY(-3px); border-color: rgba(157, 107, 255, 0.35); }
    .job-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.02rem; color: var(--text-primary); }
    .job-meta { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }
    .job-match-bar-track { background: rgba(238,241,247,0.08); border-radius: 999px; height: 6px; margin-top: 10px; overflow: hidden; }
    .job-match-bar-fill { height: 100%; background: var(--signal-gradient); border-radius: 999px; }
    .job-match-label { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; }

    /* ---------- Architecture flow ---------- */
    .flow-step {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--signal-blue);
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-primary);
    }
    .flow-arrow { text-align: center; color: var(--text-muted); font-size: 0.85rem; margin: 2px 0; }

    /* ---------- Architecture tree ---------- */
    .tree-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 20px 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.86rem;
        line-height: 2;
        color: var(--text-muted);
    }
    .tree-panel .tree-root { color: var(--text-primary); font-weight: 600; }
    .tree-panel .tree-gold { color: var(--signal-blue); font-weight: 600; }
    .tree-panel .tree-ml1 { color: var(--signal-blue); }
    .tree-panel .tree-ml2 { color: var(--signal-violet); }
    .tree-panel .tree-ml3 { color: var(--mint); }
    .tree-panel .tree-final {
        background: var(--signal-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 700;
    }

    /* ---------- Inline icons & dot indicators ---------- */
    .icon-inline { display: inline-flex; vertical-align: -3px; margin-right: 8px; }
    .icon-inline svg { width: 18px; height: 18px; }
    .dot-label { display: flex; align-items: center; gap: 8px; font-weight: 600; color: var(--text-primary); margin-bottom: 2px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .dot-mint { background: var(--mint); box-shadow: 0 0 8px rgba(53,208,160,0.6); }
    .dot-amber { background: var(--amber); box-shadow: 0 0 8px rgba(245,166,35,0.6); }

    /* ---------- GenAI chat page (Gemini-style) ---------- */
    .genai-hero {
        text-align: center;
        padding: 10px 0 22px 0;
    }
    .genai-orb {
        width: 52px; height: 52px; border-radius: 50%;
        background: var(--signal-gradient);
        margin: 0 auto 14px auto;
        box-shadow: 0 0 30px rgba(157, 107, 255, 0.45);
        animation: orb-pulse 3s ease-in-out infinite;
        display: flex; align-items: center; justify-content: center;
        color: white;
    }
    .genai-orb svg { width: 26px; height: 26px; }
    @keyframes orb-pulse {
        0%, 100% { box-shadow: 0 0 22px rgba(157, 107, 255, 0.35); }
        50% { box-shadow: 0 0 38px rgba(76, 124, 246, 0.55); }
    }
    .genai-greeting {
        font-family: 'Sora', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        background: var(--signal-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .genai-sub { color: var(--text-muted); font-size: 0.88rem; margin-top: 4px; }

    .suggestion-chip button {
        width: 100%;
        text-align: left !important;
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        padding: 12px 14px !important;
    }
    .suggestion-chip button:hover {
        border-color: var(--signal-violet) !important;
    }

    [data-testid="stChatMessage"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 4px 6px;
        margin-bottom: 10px;
        animation: msg-in 0.25s ease;
    }
    @keyframes msg-in {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Pipeline diagram (Project Overview) ---------- */
    .pipeline-wrap {
        display: flex;
        align-items: stretch;
        gap: 0;
        overflow-x: auto;
        padding: 6px 2px 14px 2px;
    }
    .pipeline-node {
        flex: 1 1 0;
        min-width: 168px;
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px 16px;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .pipeline-node:hover {
        transform: translateY(-4px);
        border-color: rgba(157, 107, 255, 0.4);
    }
    .pipeline-node .pn-icon {
        width: 36px; height: 36px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        background: var(--signal-gradient);
        color: white;
        margin-bottom: 12px;
    }
    .pipeline-node .pn-icon svg { width: 19px; height: 19px; }
    .pipeline-node .pn-stage {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .pipeline-node .pn-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    .pipeline-node .pn-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        line-height: 1.4;
    }
    .pipeline-node.pn-final .pn-icon { background: var(--mint); }
    .pipeline-node.pn-final { border-color: rgba(53, 208, 160, 0.35); }
    .pipeline-connector {
        flex: 0 0 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--signal-violet);
        opacity: 0.65;
    }
    .pipeline-connector svg { width: 20px; height: 20px; }
    .ml-sub-row {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 10px;
    }
    .ml-sub-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        padding: 4px 8px;
        border-radius: 8px;
        background: rgba(76, 124, 246, 0.1);
        border: 1px solid rgba(76, 124, 246, 0.25);
        color: var(--text-primary);
    }

    /* ---------- Insight list (point-to-point findings) ---------- */
    .insight-list { margin: 4px 0 4px 0; }
    .insight-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 9px 0;
        border-bottom: 1px solid var(--border-subtle);
        font-size: 0.86rem;
        color: var(--text-primary);
    }
    .insight-item:last-child { border-bottom: none; }
    .insight-item .ii-icon {
        flex: 0 0 18px;
        margin-top: 2px;
        color: var(--signal-violet);
    }
    .insight-item .ii-icon svg { width: 16px; height: 16px; }
    .insight-item b { color: var(--text-primary); }
    .insight-item .ii-text { color: var(--text-muted); }

    /* ---------- Step badge (Career Advisor input flow) ---------- */
    .step-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.86rem;
        color: var(--text-primary);
        margin-bottom: 10px;
    }
    .step-badge .sb-num {
        width: 22px; height: 22px;
        border-radius: 50%;
        background: var(--signal-gradient);
        color: white;
        font-size: 0.72rem;
        display: flex; align-items: center; justify-content: center;
    }

    /* ---------- Rank badge for job cards ---------- */
    .job-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px; height: 24px;
        border-radius: 7px;
        background: rgba(157, 107, 255, 0.14);
        border: 1px solid rgba(157, 107, 255, 0.3);
        color: var(--signal-violet);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        margin-right: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
CHART_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#eef1f7"),
    title_font=dict(family="Sora, sans-serif", size=15, color="#eef1f7"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=48, l=10, r=10, b=10),
)
SEQUENTIAL_PALETTE = ["#4c7cf6", "#9d6bff", "#35d0a0", "#f5a623", "#6fb8ff", "#c792ea"]

# ------------------------------------------------------------
# Professional line icons (inline SVG, stroke=currentColor) —
# used in place of emoji throughout the app.
# ------------------------------------------------------------
ICON_TARGET = '<span class="icon-inline"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg></span>'
ICON_COMPASS = '<span class="icon-inline"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polygon points="15 9 12 15 9 15 12 9 15 9"/></svg></span>'
ICON_BAR_CHART = '<span class="icon-inline"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="20" x2="6" y2="12"/><line x1="12" y1="20" x2="12" y2="6"/><line x1="18" y1="20" x2="18" y2="15"/></svg></span>'
ICON_SPARKLES = '<span class="icon-inline"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18"/></svg></span>'
ICON_GLOBE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>'
ICON_LAYERS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 21 7 12 12 3 7 12 2"/><polyline points="3 12 12 17 21 12"/><polyline points="3 17 12 22 21 17"/></svg>'
ICON_DATABASE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></svg>'
ICON_CPU = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="10" y="10" width="4" height="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M6 2v2M18 2v2M6 20v2M18 20v2M2 6h2M2 18h2M20 6h2M20 18h2"/></svg>'
ICON_CHAT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>'
ICON_CHECK_CIRCLE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/></svg>'
ICON_ALERT_CIRCLE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4.5"/><circle cx="12" cy="16" r="0.6" fill="currentColor" stroke="none"/></svg>'
ICON_BRIEFCASE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M3 12h18"/></svg>'
ICON_WARNING_TRIANGLE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5 21 20H3L12 3.5z"/><path d="M12 9.5v4.2"/><circle cx="12" cy="16.8" r="0.6" fill="currentColor" stroke="none"/></svg>'
ICON_STEP = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h13M13 6l6 6-6 6"/></svg>'


def card_container():
    """A bordered container that degrades gracefully on older Streamlit versions."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def stream_answer(placeholder, text, delay=0.018):
    """Renders text word-by-word to simulate an LLM streaming response."""
    words = text.split(" ")
    shown = ""
    for i, word in enumerate(words):
        shown += word + (" " if i < len(words) - 1 else "")
        placeholder.markdown(shown + " ▌")
        time.sleep(delay)
    placeholder.markdown(shown)


# ============================================================
# UI HELPER COMPONENTS
# ============================================================

def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def content_header(icon_svg, text):
    st.markdown(
        f'<div class="section-label"><span class="icon-inline" style="vertical-align:-4px;">{icon_svg}</span>{text}</div>',
        unsafe_allow_html=True
    )


def styled_bar(df, x_col, y_col, title, orientation="h", color_seq=None, height=380):
    color_seq = color_seq or ["#4c7cf6"]
    if orientation == "h":
        fig = px.bar(df, x=y_col, y=x_col, orientation="h", text_auto=".2s",
                     color_discrete_sequence=color_seq, template=PLOTLY_TEMPLATE, title=title)
        fig.update_layout(yaxis_title="", xaxis_title=y_col.replace("_", " ").title())
    else:
        fig = px.bar(df, x=x_col, y=y_col, text_auto=".2s",
                     color_discrete_sequence=color_seq, template=PLOTLY_TEMPLATE, title=title)
        fig.update_layout(xaxis_title="", yaxis_title=y_col.replace("_", " ").title())
    fig.update_traces(marker_color=color_seq[0])
    fig.update_layout(**CHART_LAYOUT_DEFAULTS, height=height, showlegend=False)
    return fig


def skill_chips(skills, kind="available", limit=15):
    cls = "chip-available" if kind == "available" else "chip-missing"
    chips_html = "".join(f'<span class="chip {cls}">{s}</span>' for s in skills[:limit])
    st.markdown(f'<div class="chip-row">{chips_html}</div>', unsafe_allow_html=True)


def job_card(title, company, location, experience, salary, match_score, job_url=None, rank=None):
    link_html = f'<a href="{job_url}" target="_blank" style="color:var(--signal-violet); font-size:0.78rem;">View posting →</a>' if job_url and str(job_url) != "nan" else ""
    rank_html = f'<span class="job-rank">{rank}</span>' if rank is not None else ""
    st.markdown(f"""
    <div class="job-card">
        <div class="job-title">{rank_html}{title}</div>
        <div class="job-meta">{company} · {location} · {experience} · {salary}</div>
        <div class="job-match-bar-track"><div class="job-match-bar-fill" style="width:{match_score}%;"></div></div>
        <div class="job-match-label">MATCH SCORE {match_score:.1f}%</div>
        {link_html}
    </div>
    """, unsafe_allow_html=True)


def insight_list(items):
    """Renders a clean, point-to-point bullet list for findings/answers.
    items: list of (icon_svg, bold_lead, rest_text) tuples.
    """
    rows = []
    for icon_svg, lead, rest in items:
        rows.append(f"""
        <div class="insight-item">
            <span class="ii-icon">{icon_svg}</span>
            <span><b>{lead}</b> <span class="ii-text">{rest}</span></span>
        </div>
        """)
    st.markdown(f'<div class="insight-list">{"".join(rows)}</div>', unsafe_allow_html=True)


def step_badge(number, text):
    st.markdown(
        f'<div class="step-badge"><span class="sb-num">{number}</span>{text}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# UTILS & MODEL LOADERS
# ============================================================

def load_pickle(path):
    if not Path(path).exists():
        return None

    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data
def load_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_gold_data():
    data = {
        "jobs": load_csv(JOBS_PATH),
        "skill_demand": load_csv(SKILL_DEMAND_PATH),
        "role_demand": load_csv(ROLE_DEMAND_PATH),
        "company_demand": load_csv(COMPANY_DEMAND_PATH),
        "location_demand": load_csv(LOCATION_DEMAND_PATH),
        "salary_insights": load_csv(SALARY_INSIGHTS_PATH),
        "source_quality": load_csv(SOURCE_QUALITY_PATH),
    }
    return data


@st.cache_resource
def load_role_classifier_artifacts():
    """
    Auto-detects ML1 role classifier model/vectorizer/label encoder
    from models folder.
    """
    artifacts = {
        "model": None,
        "vectorizer": None,
        "label_encoder": None,
        "model_path": None,
        "vectorizer_path": None,
        "label_encoder_path": None,
    }

    pkl_files = list(MODEL_DIR.rglob("*.pkl"))

    # Avoid loading ML3 recommender files as ML1 artifacts
    pkl_files = [
        p for p in pkl_files
        if "job_recommender" not in str(p).replace("\\", "/")
    ]

    for path in pkl_files:
        obj = load_pickle(path)
        if obj is None:
            continue

        if hasattr(obj, "classes_") and hasattr(obj, "inverse_transform"):
            artifacts["label_encoder"] = obj
            artifacts["label_encoder_path"] = str(path)

        elif hasattr(obj, "vocabulary_") and hasattr(obj, "transform"):
            artifacts["vectorizer"] = obj
            artifacts["vectorizer_path"] = str(path)

        elif hasattr(obj, "predict"):
            artifacts["model"] = obj
            artifacts["model_path"] = str(path)

    return artifacts


@st.cache_resource
def load_recommender_artifacts():
    vectorizer = load_pickle(RECOMMENDER_VECTORIZER_PATH)
    vectors = load_pickle(RECOMMENDER_VECTORS_PATH)
    jobs = load_csv(RECOMMENDER_JOBS_PATH)

    return {
        "vectorizer": vectorizer,
        "vectors": vectors,
        "jobs": jobs,
    }


@st.cache_resource
def get_ai_service():
    """
    Loads and caches the AIService (Groq + Guardrails).
    """
    try:
        from genai.ai_service import AIService
        return AIService()
    except Exception as e:
        st.error(f"Failed to initialize AI Service: {e}")
        return None


def parse_skill_list(value):
    if pd.isna(value):
        return []

    value = str(value).lower().strip()

    if value == "":
        return []

    value = value.replace("[", "").replace("]", "")
    value = value.replace("'", "").replace('"', "")

    parts = re.split(r",|\||;", value)
    clean = []

    for item in parts:
        item = item.strip()
        if item:
            clean.append(item)

    return clean


def normalize_user_skills(user_text, market_skills=None):
    text = str(user_text).lower()

    found = set()

    if market_skills is not None:
        for skill in market_skills:
            skill = str(skill).lower().strip()
            if not skill:
                continue

            pattern = r"\b" + re.escape(skill).replace("\\ ", r"\s+") + r"\b"
            if re.search(pattern, text):
                found.add(skill)

    tokens = re.split(r"[,|\n|\s]+", text)
    for token in tokens:
        token = token.strip()
        if token:
            found.add(token)

    return found


def predict_role(user_skills_text):
    artifacts = load_role_classifier_artifacts()
    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    label_encoder = artifacts["label_encoder"]

    if model is None:
        return None, artifacts

    input_text = str(user_skills_text).strip()

    try:
        pred = model.predict([input_text])
    except Exception:
        if vectorizer is None:
            return None, artifacts

        X = vectorizer.transform([input_text])
        pred = model.predict(X)

    role = pred[0]

    try:
        if label_encoder is not None and not isinstance(role, str):
            role = label_encoder.inverse_transform([role])[0]
    except Exception:
        pass

    return str(role), artifacts


def calculate_skill_gap(jobs, skill_demand, predicted_role, user_skills_text):
    if jobs.empty:
        return {
            "profile_match": 0,
            "available": pd.DataFrame(),
            "missing": pd.DataFrame(),
            "role_market_jobs": 0,
            "top_role_skills": pd.DataFrame(),
        }

    skill_col = None

    if "extracted_skills" in jobs.columns:
        skill_col = "extracted_skills"
    elif "skills" in jobs.columns:
        skill_col = "skills"

    if skill_col is None:
        return {
            "profile_match": 0,
            "available": pd.DataFrame(),
            "missing": pd.DataFrame(),
            "role_market_jobs": 0,
            "top_role_skills": pd.DataFrame(),
        }

    if "role_category" in jobs.columns and predicted_role:
        role_jobs = jobs[jobs["role_category"].astype(str) == str(predicted_role)]
        if role_jobs.empty:
            role_jobs = jobs.copy()
    else:
        role_jobs = jobs.copy()

    all_skills = []

    for value in role_jobs[skill_col].dropna():
        all_skills.extend(parse_skill_list(value))

    if len(all_skills) == 0:
        return {
            "profile_match": 0,
            "available": pd.DataFrame(),
            "missing": pd.DataFrame(),
            "role_market_jobs": len(role_jobs),
            "top_role_skills": pd.DataFrame(),
        }

    skill_counts = (
        pd.Series(all_skills)
        .value_counts()
        .reset_index()
    )

    skill_counts.columns = ["skill", "job_count"]

    skill_counts["demand_percent"] = round(
        (skill_counts["job_count"] / len(role_jobs)) * 100, 2
    )

    top_role_skills = skill_counts.head(20).copy()

    market_skills = skill_counts["skill"].tolist()
    user_skills = normalize_user_skills(user_skills_text, market_skills)

    def is_available(skill):
        skill = str(skill).lower()
        return skill in user_skills or skill in str(user_skills_text).lower()

    top_role_skills["status"] = top_role_skills["skill"].apply(
        lambda x: "Available" if is_available(x) else "Missing"
    )

    available = top_role_skills[top_role_skills["status"] == "Available"].copy()
    missing = top_role_skills[top_role_skills["status"] == "Missing"].copy()

    if top_role_skills["job_count"].sum() > 0:
        profile_match = round(
            (available["job_count"].sum() / top_role_skills["job_count"].sum()) * 100,
            2
        )
    else:
        profile_match = 0

    return {
        "profile_match": profile_match,
        "available": available,
        "missing": missing,
        "role_market_jobs": len(role_jobs),
        "top_role_skills": top_role_skills,
    }


def recommend_jobs(user_skills_text, predicted_role=None, top_n=5):
    rec = load_recommender_artifacts()

    vectorizer = rec["vectorizer"]
    vectors = rec["vectors"]
    rec_jobs = rec["jobs"]

    if vectorizer is None or vectors is None or rec_jobs.empty or cosine_similarity is None:
        return pd.DataFrame()

    query = str(user_skills_text)

    if predicted_role:
        query = f"{predicted_role} {query}"

    try:
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, vectors).ravel()
        top_idx = np.argsort(scores)[::-1][:top_n]

        result = rec_jobs.iloc[top_idx].copy()
        result["match_score"] = [round(float(scores[i]) * 100, 2) for i in top_idx]

        display_cols = [
            "job_title",
            "company",
            "location",
            "experience",
            "salary",
            "role_category",
            "skills",
            "match_score",
            "job_url"
        ]

        available_cols = [c for c in display_cols if c in result.columns]
        return result[available_cols]

    except Exception:
        return pd.DataFrame()


def generate_market_context(data, predicted_role=None, skill_gap=None, recommendations=None):
    jobs = data["jobs"]
    skill_demand = data["skill_demand"]
    role_demand = data["role_demand"]
    company_demand = data["company_demand"]

    context = []

    if not jobs.empty:
        context.append(f"Total cleaned job records: {len(jobs)}")

    if not role_demand.empty:
        role_cols = role_demand.columns.tolist()
        if "role_category" in role_cols and "job_count" in role_cols:
            top_roles = role_demand.sort_values("job_count", ascending=False).head(7)
            context.append("Role demand:")
            for _, row in top_roles.iterrows():
                context.append(f"- {row['role_category']}: {row['job_count']} jobs")

    if not skill_demand.empty:
        if "skill" in skill_demand.columns and "job_count" in skill_demand.columns:
            top_skills = skill_demand.sort_values("job_count", ascending=False).head(10)
            context.append("Top market skills:")
            for _, row in top_skills.iterrows():
                context.append(f"- {row['skill']}: {row['job_count']} jobs")

    if not company_demand.empty:
        if "company" in company_demand.columns and "job_count" in company_demand.columns:
            top_companies = company_demand.sort_values("job_count", ascending=False).head(10)
            context.append("Top hiring companies:")
            for _, row in top_companies.iterrows():
                context.append(f"- {row['company']}: {row['job_count']} jobs")

    if predicted_role:
        context.append(f"Predicted best role for user: {predicted_role}")

    if skill_gap:
        context.append(f"Market jobs analyzed for predicted role: {skill_gap['role_market_jobs']}")
        context.append(f"Profile match: {skill_gap['profile_match']}%")

        if not skill_gap["available"].empty:
            available_skills = ", ".join(skill_gap["available"]["skill"].head(10).tolist())
            context.append(f"Available skills: {available_skills}")

        if not skill_gap["missing"].empty:
            missing_skills = ", ".join(skill_gap["missing"]["skill"].head(10).tolist())
            context.append(f"Missing skills: {missing_skills}")

    if recommendations is not None and not recommendations.empty:
        context.append("Recommended jobs:")
        for _, row in recommendations.head(5).iterrows():
            title = row.get("job_title", "Job")
            company = row.get("company", "Company")
            score = row.get("match_score", "")
            context.append(f"- {title} at {company}, match score {score}")

    return "\n".join(context)


# ============================================================
# LOAD DATA
# ============================================================

data = load_gold_data()
jobs = data["jobs"]
skill_demand = data["skill_demand"]
role_demand = data["role_demand"]
company_demand = data["company_demand"]
location_demand = data["location_demand"]
salary_insights = data["salary_insights"]
source_quality = data["source_quality"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-orb"></div>
        <div>
            <div class="sidebar-brand-text">Job Market<br/>Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Go to",
        [
            "Project Overview",
            "Market Analytics",
            "Career Advisor",
            "GenAI Assistant"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Major Project · Big Data + ML + GenAI")


# ============================================================
# PAGE 1: PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":
    st.markdown(f"""
    <div class="hero-band">
        <div class="hero-eyebrow">{ICON_COMPASS} Platform overview</div>
        <p class="hero-title">Job Market Intelligence Platform</p>
        <p class="hero-subtitle">Scraped job market data, cleaned through an ETL pipeline, powering
        a role classifier, a skill-gap analyzer, a job recommender, and a GenAI career advisor.</p>
    </div>
    """, unsafe_allow_html=True)

    if jobs.empty:
        st.error("Gold data not found. Please check data/gold_v3/jobs_gold.csv")
    else:
        total_jobs = len(jobs)
        unique_companies = jobs["company"].nunique() if "company" in jobs.columns else 0
        unique_locations = jobs["location"].nunique() if "location" in jobs.columns else 0
        sources = jobs["source"].nunique() if "source" in jobs.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Cleaned jobs", f"{total_jobs:,}", "Gold layer records")
        with c2:
            kpi_card("Companies", f"{unique_companies:,}", "Unique hiring orgs")
        with c3:
            kpi_card("Locations", f"{unique_locations:,}", "Distinct markets")
        with c4:
            kpi_card("Sources", f"{sources:,}", "Scraped platforms")

        section_label("Project architecture")

        connector_svg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h13M13 6l6 6-6 6"/></svg>'

        pipeline_html = f"""
        <div class="pipeline-wrap">
            <div class="pipeline-node">
                <div class="pn-icon">{ICON_GLOBE}</div>
                <div class="pn-stage">Stage 01</div>
                <div class="pn-title">Web Scraping</div>
                <div class="pn-sub">Multi-source job postings collected into a raw landing zone.</div>
            </div>
            <div class="pipeline-connector">{connector_svg}</div>
            <div class="pipeline-node">
                <div class="pn-icon">{ICON_LAYERS}</div>
                <div class="pn-stage">Stage 02</div>
                <div class="pn-title">Raw → Silver</div>
                <div class="pn-sub">Deduplication, text cleaning, and schema normalization.</div>
            </div>
            <div class="pipeline-connector">{connector_svg}</div>
            <div class="pipeline-node">
                <div class="pn-icon">{ICON_DATABASE}</div>
                <div class="pn-stage">Stage 03</div>
                <div class="pn-title">Gold Analytical Layer</div>
                <div class="pn-sub">Aggregated, analysis-ready tables that feed every model below.</div>
            </div>
            <div class="pipeline-connector">{connector_svg}</div>
            <div class="pipeline-node">
                <div class="pn-icon">{ICON_CPU}</div>
                <div class="pn-stage">Stage 04</div>
                <div class="pn-title">Machine Learning Layer</div>
                <div class="pn-sub">Three models trained on the gold layer.</div>
                <div class="ml-sub-row">
                    <span class="ml-sub-chip">ML1 · Role Classifier</span>
                    <span class="ml-sub-chip">ML2 · Skill Gap Analyzer</span>
                    <span class="ml-sub-chip">ML3 · Job Recommender</span>
                </div>
            </div>
            <div class="pipeline-connector">{connector_svg}</div>
            <div class="pipeline-node pn-final">
                <div class="pn-icon">{ICON_CHAT}</div>
                <div class="pn-stage">Stage 05</div>
                <div class="pn-title">GenAI Career Advisor</div>
                <div class="pn-sub">Groq-powered chat, grounded in the gold layer with prompt guardrails.</div>
            </div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)

        section_label("Sample gold records")
        preview_cols = [c for c in jobs.columns if c not in ("experience", "salary")]
        st.dataframe(jobs[preview_cols].head(20), use_container_width=True)


# ============================================================
# PAGE 2: MARKET ANALYTICS
# ============================================================

elif page == "Market Analytics":
    st.markdown(f"""
    <div class="hero-band">
        <div class="hero-eyebrow">{ICON_BAR_CHART} Market analytics</div>
        <p class="hero-title">Where the demand actually is</p>
        <p class="hero-subtitle">Roles, skills, companies, and locations ranked by real job volume from the gold layer.</p>
    </div>
    """, unsafe_allow_html=True)

    if jobs.empty:
        st.error("Gold data not found.")
    else:
        col_a, col_b = st.columns([1.3, 1])

        with col_a:
            section_label("Role demand")
            if not role_demand.empty:
                if "job_count" in role_demand.columns:
                    rd = role_demand.sort_values("job_count", ascending=False).head(12)
                    index_col = "role_category" if "role_category" in rd.columns else rd.columns[0]
                    fig = styled_bar(rd.sort_values("job_count"), index_col, "job_count",
                                      "Top roles by job count", color_seq=["#4c7cf6"], height=420)
                    st.plotly_chart(fig, use_container_width=True)

                    fig_tree = px.treemap(
                        role_demand, path=[index_col], values="job_count",
                        color="job_count", color_continuous_scale=["#171a26", "#4c7cf6", "#9d6bff"],
                        template=PLOTLY_TEMPLATE, title="Role Demand Share — All Roles"
                    )
                    fig_tree.update_layout(**CHART_LAYOUT_DEFAULTS, height=380, coloraxis_showscale=False)
                    st.plotly_chart(fig_tree, use_container_width=True)
                else:
                    st.dataframe(role_demand, use_container_width=True)
            else:
                st.info("Role demand data not available.")

        with col_b:
            section_label("Source-wise job count")
            if "source" in jobs.columns:
                source_count = jobs["source"].value_counts().reset_index()
                source_count.columns = ["source", "job_count"]
                fig_src = go.Figure(data=[go.Pie(
                    labels=source_count["source"], values=source_count["job_count"], hole=0.62,
                    marker=dict(colors=SEQUENTIAL_PALETTE, line=dict(color="#0f111a", width=2)),
                    textinfo="percent",
                    hovertemplate="<b>%{label}</b><br>%{value:,} jobs<br>%{percent}<extra></extra>"
                )])
                fig_src.update_layout(**CHART_LAYOUT_DEFAULTS, height=420, title="Job volume by source")
                st.plotly_chart(fig_src, use_container_width=True)
            else:
                st.info("Source column not available.")

        section_label("Top skills in demand")
        if not skill_demand.empty:
            if "job_count" in skill_demand.columns:
                sd = skill_demand.sort_values("job_count", ascending=False).head(20)
                index_col = "skill" if "skill" in sd.columns else sd.columns[0]
                fig_sd = styled_bar(sd.sort_values("job_count"), index_col, "job_count",
                                     "Most requested skills across all roles", color_seq=["#9d6bff"], height=520)
                st.plotly_chart(fig_sd, use_container_width=True)
                with st.expander("View full skill demand table"):
                    st.dataframe(sd, use_container_width=True)
            else:
                st.dataframe(skill_demand, use_container_width=True)
        else:
            st.info("Skill demand data not available.")

        col_c, col_d = st.columns(2)
        with col_c:
            section_label("Top hiring companies")
            if not company_demand.empty:
                if "job_count" in company_demand.columns:
                    cd = company_demand.sort_values("job_count", ascending=False).head(15)
                    index_col = "company" if "company" in cd.columns else cd.columns[0]
                    fig_cd = styled_bar(cd.sort_values("job_count"), index_col, "job_count",
                                         "Top companies by open roles", color_seq=["#35d0a0"], height=460)
                    st.plotly_chart(fig_cd, use_container_width=True)
                    with st.expander("View full company demand table"):
                        st.dataframe(cd, use_container_width=True)
                else:
                    st.dataframe(company_demand.head(20), use_container_width=True)
            else:
                st.info("Company demand data not available.")

        with col_d:
            section_label("Top locations")
            if not location_demand.empty:
                if "job_count" in location_demand.columns:
                    ld = location_demand.sort_values("job_count", ascending=False).head(15)
                    index_col = "location" if "location" in ld.columns else ld.columns[0]
                    fig_ld = styled_bar(ld.sort_values("job_count"), index_col, "job_count",
                                         "Top hiring locations", color_seq=["#f5a623"], height=460)
                    st.plotly_chart(fig_ld, use_container_width=True)
                    with st.expander("View full location demand table"):
                        st.dataframe(ld, use_container_width=True)
                else:
                    st.dataframe(location_demand.head(20), use_container_width=True)
            else:
                st.info("Location demand data not available.")

        section_label("Salary insights")
        if not salary_insights.empty:
            si = salary_insights.copy()
            numeric_cols = si.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in si.columns if c not in numeric_cols]
            cat_col = cat_cols[0] if cat_cols else None
            min_col = next((c for c in numeric_cols if "min" in c.lower()), None)
            max_col = next((c for c in numeric_cols if "max" in c.lower()), None)
            avg_col = next((c for c in numeric_cols if any(k in c.lower() for k in ["avg", "mean", "median"])), None)

            if cat_col and min_col and max_col:
                plot_df = si[[cat_col, min_col, max_col] + ([avg_col] if avg_col else [])].dropna().copy()
                plot_df["_mid"] = plot_df[avg_col] if avg_col else (plot_df[min_col] + plot_df[max_col]) / 2
                plot_df = plot_df.sort_values("_mid")

                fig_sal = go.Figure()
                for _, row in plot_df.iterrows():
                    fig_sal.add_trace(go.Scatter(
                        x=[row[min_col], row[max_col]], y=[row[cat_col], row[cat_col]],
                        mode="lines", line=dict(color="#9d6bff", width=6), opacity=0.35,
                        showlegend=False, hoverinfo="skip"
                    ))
                    fig_sal.add_trace(go.Scatter(
                        x=[row["_mid"]], y=[row[cat_col]], mode="markers",
                        marker=dict(color="#4c7cf6", size=13, line=dict(color="#0f111a", width=2)),
                        showlegend=False,
                        hovertemplate=f"<b>{row[cat_col]}</b><br>Range: {row[min_col]:,.0f}–{row[max_col]:,.0f}<extra></extra>"
                    ))
                fig_sal.update_layout(**CHART_LAYOUT_DEFAULTS, height=460, title="Salary Range by Role",
                                       xaxis_title="Salary", yaxis_title="", template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_sal, use_container_width=True)

            elif cat_col and avg_col:
                plot_df = si[[cat_col, avg_col]].dropna().sort_values(avg_col)
                fig_sal = styled_bar(plot_df, cat_col, avg_col, "Average Salary by Role",
                                      color_seq=["#4c7cf6"], height=440)
                st.plotly_chart(fig_sal, use_container_width=True)

            elif cat_col and numeric_cols:
                val_col = numeric_cols[0]
                plot_df = si[[cat_col, val_col]].dropna().sort_values(val_col)
                fig_sal = styled_bar(plot_df, cat_col, val_col,
                                      f"{val_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                                      color_seq=["#4c7cf6"], height=440)
                st.plotly_chart(fig_sal, use_container_width=True)
            else:
                st.dataframe(salary_insights, use_container_width=True)
                st.caption("Couldn't detect a category + numeric column pair to chart — showing raw data.")
        else:
            st.info("Salary insights data not available.")

        section_label("Source quality")
        if not source_quality.empty:
            sq = source_quality.copy()
            sq_numeric = sq.select_dtypes(include=[np.number]).columns.tolist()
            sq_cat_cols = [c for c in sq.columns if c not in sq_numeric]
            sq_cat_col = sq_cat_cols[0] if sq_cat_cols else None

            if sq_cat_col and len(sq_numeric) == 1:
                plot_df = sq[[sq_cat_col, sq_numeric[0]]].dropna().sort_values(sq_numeric[0])
                fig_sq = styled_bar(plot_df, sq_cat_col, sq_numeric[0],
                                     f"{sq_numeric[0].replace('_', ' ').title()} by Source",
                                     color_seq=["#35d0a0"], height=420)
                st.plotly_chart(fig_sq, use_container_width=True)
            elif sq_cat_col and len(sq_numeric) > 1:
                melt_df = sq.melt(id_vars=[sq_cat_col], value_vars=sq_numeric, var_name="metric", value_name="value")
                fig_sq = px.bar(melt_df, x=sq_cat_col, y="value", color="metric", barmode="group",
                                 color_discrete_sequence=SEQUENTIAL_PALETTE, template=PLOTLY_TEMPLATE,
                                 title="Source Quality Metrics by Source")
                fig_sq.update_layout(**CHART_LAYOUT_DEFAULTS, height=440, xaxis_title="", yaxis_title="Score")
                st.plotly_chart(fig_sq, use_container_width=True)
            else:
                st.dataframe(source_quality, use_container_width=True)
                st.caption("Couldn't detect chartable columns — showing raw data.")
        else:
            st.info("Source quality data not available.")


# ============================================================
# PAGE 3: CAREER ADVISOR
# ============================================================

elif page == "Career Advisor":
    st.markdown(f"""
    <div class="hero-band">
        <div class="hero-eyebrow">{ICON_TARGET} Career advisor · ML1 + ML2 + ML3</div>
        <p class="hero-title">Find your role, close your gap, get matched</p>
        <p class="hero-subtitle">Enter your skills once — three models run in sequence: a role classifier predicts
        your best-fit role, a skill-gap analyzer scores you against real market demand, and a recommender
        surfaces the jobs you're most likely to land.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- How it works strip ----------------
    hw1, hw2, hw3 = st.columns(3)
    with hw1:
        st.markdown(f"""
        <div class="pipeline-node" style="min-width:auto;">
            <div class="pn-icon">{ICON_TARGET}</div>
            <div class="pn-stage">ML1</div>
            <div class="pn-title">Role Match</div>
            <div class="pn-sub">Classifies your skill set into the single best-fit role category.</div>
        </div>
        """, unsafe_allow_html=True)
    with hw2:
        st.markdown(f"""
        <div class="pipeline-node" style="min-width:auto;">
            <div class="pn-icon">{ICON_ALERT_CIRCLE}</div>
            <div class="pn-stage">ML2</div>
            <div class="pn-title">Skill Gap</div>
            <div class="pn-sub">Compares your skills against real postings for that role.</div>
        </div>
        """, unsafe_allow_html=True)
    with hw3:
        st.markdown(f"""
        <div class="pipeline-node pn-final" style="min-width:auto;">
            <div class="pn-icon">{ICON_BRIEFCASE}</div>
            <div class="pn-stage">ML3</div>
            <div class="pn-title">Recommended Jobs</div>
            <div class="pn-sub">Ranks live postings by similarity to your profile.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------------- Input panel ----------------
    with card_container():
        step_badge(1, "Tell us about your skills")
        user_skills = st.text_area(
            "Enter your skills",
            value="python sql pandas machine learning",
            height=100,
            help="Example: python sql pandas pyspark airflow aws",
            label_visibility="collapsed"
        )

        col_num, col_button = st.columns([1, 2])
        with col_num:
            top_n = st.number_input(
                "Number of job recommendations",
                min_value=1, max_value=20, value=5, step=1,
                help="How many matching jobs ML3 should return"
            )
        with col_button:
            st.write("")
            analyze_clicked = st.button("Analyze my career path", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_skills.strip():
            st.warning("Please enter at least one skill.")
        else:
            predicted_role, role_artifacts = predict_role(user_skills)

            if predicted_role is None:
                st.error("ML1 Role Classifier model not found or not loaded.")
                st.write("Check your saved ML1 model files inside the models/ folder.")
                with st.expander("Detected artifacts"):
                    st.json(role_artifacts)
            else:
                skill_gap = calculate_skill_gap(
                    jobs=jobs,
                    skill_demand=skill_demand,
                    predicted_role=predicted_role,
                    user_skills_text=user_skills
                )

                recommendations = recommend_jobs(
                    user_skills_text=user_skills,
                    predicted_role=predicted_role,
                    top_n=int(top_n)
                )

                st.session_state["last_user_skills"] = user_skills
                st.session_state["last_predicted_role"] = predicted_role
                st.session_state["last_skill_gap"] = skill_gap
                st.session_state["last_recommendations"] = recommendations

                # ---------------- Result summary strip ----------------
                st.write("")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    kpi_card("Predicted role", predicted_role, "ML1 · Role Classifier")
                with s2:
                    kpi_card("Profile match", f"{skill_gap['profile_match']}%", "ML2 · Skill Gap Analyzer")
                with s3:
                    kpi_card("Market jobs analyzed", f"{skill_gap['role_market_jobs']:,}", "For this role category")
                with s4:
                    kpi_card("Jobs recommended", f"{len(recommendations)}", "ML3 · Job Recommender")

                st.write("")
                tab_role, tab_gap, tab_jobs = st.tabs(["Role Match", "Skill Gap", "Recommended Jobs"])

                # ---------------- TAB: ROLE MATCH ----------------
                with tab_role:
                    content_header(ICON_TARGET, "Role match — point-to-point")
                    insight_list([
                        (ICON_CHECK_CIRCLE, "Predicted role:",
                         f"{predicted_role}, the closest match to your entered skills."),
                        (ICON_DATABASE, "Market coverage:",
                         f"{skill_gap['role_market_jobs']:,} live postings analyzed for this role category."),
                        (ICON_TARGET, "Profile match score:",
                         f"{skill_gap['profile_match']}% overlap with the top skills this role demands."),
                    ])

                    st.write("")
                    col_role, col_gauge = st.columns([1, 1])
                    with col_role:
                        kpi_card("Predicted role", predicted_role)
                        kpi_card("Market jobs analyzed", f"{skill_gap['role_market_jobs']:,}", "For this role category")

                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=skill_gap["profile_match"],
                            number={"suffix": "%", "font": {"family": "Sora, sans-serif", "size": 40, "color": "#eef1f7"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#8d94a8"},
                                "bar": {"color": "#4c7cf6"},
                                "bgcolor": "rgba(0,0,0,0)",
                                "borderwidth": 0,
                                "steps": [
                                    {"range": [0, 40], "color": "rgba(245,166,35,0.18)"},
                                    {"range": [40, 70], "color": "rgba(76,124,246,0.16)"},
                                    {"range": [70, 100], "color": "rgba(53,208,160,0.18)"},
                                ],
                            },
                            title={"text": f"Profile Match — {predicted_role}", "font": {"family": "Sora, sans-serif", "size": 14, "color": "#eef1f7"}}
                        ))
                        layout = dict(CHART_LAYOUT_DEFAULTS)
                        layout["height"] = 260
                        layout["margin"] = dict(t=40, l=20, r=20, b=10)

                        fig_gauge.update_layout(**layout)
                        st.plotly_chart(fig_gauge, use_container_width=True)

                # ---------------- TAB: SKILL GAP ----------------
                with tab_gap:
                    content_header(ICON_ALERT_CIRCLE, "Skill gap — point-to-point")

                    n_available = len(skill_gap["available"])
                    n_missing = len(skill_gap["missing"])
                    n_total = len(skill_gap["top_role_skills"])
                    top_missing = (
                        skill_gap["missing"].sort_values("job_count", ascending=False)["skill"].iloc[0]
                        if not skill_gap["missing"].empty else None
                    )

                    insight_items = [
                        (ICON_CHECK_CIRCLE, f"{n_available} of {n_total} top in-demand skills",
                         "for this role are already in your profile."),
                        (ICON_ALERT_CIRCLE, f"{n_missing} skills are missing",
                         "based on frequency across real postings for this role."),
                    ]
                    if top_missing:
                        insight_items.append(
                            (ICON_ALERT_CIRCLE, "Highest-impact gap:",
                             f"\"{top_missing}\" appears most often among your missing skills — learn this first.")
                        )
                    insight_list(insight_items)

                    st.write("")
                    if not skill_gap["top_role_skills"].empty:
                        gap_df = skill_gap["top_role_skills"].sort_values("job_count")
                        color_map = {"Available": "#35d0a0", "Missing": "#f5a623"}
                        fig_gap = px.bar(
                            gap_df, x="job_count", y="skill", color="status", orientation="h",
                            color_discrete_map=color_map, template=PLOTLY_TEMPLATE,
                            title=f"Skill Coverage for {predicted_role} — What You Have vs. What The Market Wants"
                        )
                        fig_gap.update_layout(**CHART_LAYOUT_DEFAULTS, height=460, yaxis_title="", xaxis_title="Job count")
                        st.plotly_chart(fig_gap, use_container_width=True)
                    else:
                        st.info("No strong matching skill data found for this role.")

                    col_avail, col_missing = st.columns(2)
                    with col_avail:
                        st.markdown('<div class="dot-label"><span class="dot dot-mint"></span> Skills you already have</div>', unsafe_allow_html=True)
                        if not skill_gap["available"].empty:
                            skill_chips(skill_gap["available"]["skill"].tolist(), kind="available")
                        else:
                            st.caption("No strong matching skills found in the top role skills.")
                    with col_missing:
                        st.markdown('<div class="dot-label"><span class="dot dot-amber"></span> Skills worth learning next</div>', unsafe_allow_html=True)
                        if not skill_gap["missing"].empty:
                            skill_chips(skill_gap["missing"]["skill"].tolist(), kind="missing")
                        else:
                            st.caption("No major missing skill found in top role skills.")

                # ---------------- TAB: RECOMMENDED JOBS ----------------
                with tab_jobs:
                    content_header(ICON_BRIEFCASE, "Recommended jobs — point-to-point")

                    if not recommendations.empty:
                        top_row = recommendations.iloc[0]
                        insight_list([
                            (ICON_BRIEFCASE, f"Showing top {len(recommendations)} roles,",
                             f"ranked by similarity to your skills and predicted role ({predicted_role})."),
                            (ICON_CHECK_CIRCLE, "Best match:",
                             f"{top_row.get('job_title', 'Job')} at {top_row.get('company', '—')} — "
                             f"{float(top_row.get('match_score', 0)):.1f}% match."),
                        ])
                        st.write("")
                        for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
                            job_card(
                                title=row.get("job_title", "Job"),
                                company=row.get("company", "—"),
                                location=row.get("location", "—"),
                                experience=row.get("experience", "—"),
                                salary=row.get("salary", "—"),
                                match_score=float(row.get("match_score", 0)),
                                job_url=row.get("job_url", None),
                                rank=i,
                            )
                        with st.expander("View as table"):
                            st.dataframe(recommendations, use_container_width=True)
                    else:
                        insight_list([
                            (ICON_ALERT_CIRCLE, "No recommendations available.",
                             "ML3 recommender artifacts not found or could not be loaded."),
                        ])
                        st.warning(
                            "ML3 recommender artifacts not found or could not be loaded. "
                            "Check the models/job_recommender folder."
                        )

                with st.expander("Model artifact status (debug)"):
                    st.json(role_artifacts)


# ============================================================
# PAGE 4: GENAI ASSISTANT (GROQ + GUARDRAILS) — GEMINI-STYLE CHAT
# ============================================================

elif page == "GenAI Assistant":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    use_last_analysis = True
    pending_question = None

    if len(st.session_state.chat_history) == 0:
        st.markdown(f"""
        <div class="genai-hero">
            <div class="genai-orb">{ICON_SPARKLES}</div>
            <div class="genai-greeting">How can I help with your career today?</div>
            <div class="genai-sub">Powered by Groq AI, grounded in your real job market data, with prompt guardrails.</div>
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            "I know Python, SQL and Pandas. What data role should I target and what should I learn next?",
            "Which skills give me the fastest path into a Data Analyst role?",
            "Compare demand for Data Engineer vs Data Scientist roles in this market.",
            "Give me a 30-day roadmap to become job-ready based on current market demand.",
        ]
        sc1, sc2 = st.columns(2)
        cols = [sc1, sc2, sc1, sc2]
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                st.markdown('<div class="suggestion-chip">', unsafe_allow_html=True)
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    pending_question = suggestion
                st.markdown('</div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask a career question…")
    question = pending_question or typed_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        ai_service = get_ai_service()

        with st.chat_message("assistant"):
            if ai_service is None:
                answer = "AIService is not initialized properly. Ensure GROQ_API_KEY is present in your .env file."
                st.error(answer)
            else:
                predicted_role = None
                skill_gap = None
                recommendations = None

                if use_last_analysis:
                    predicted_role = st.session_state.get("last_predicted_role")
                    skill_gap = st.session_state.get("last_skill_gap")
                    recommendations = st.session_state.get("last_recommendations")

                market_context = generate_market_context(
                    data=data,
                    predicted_role=predicted_role,
                    skill_gap=skill_gap,
                    recommendations=recommendations
                )

                system_prompt = (
                    "You are a Job Market Intelligence Career Advisor.\n"
                    "Rules:\n"
                    "1. Answer only using the given job market context.\n"
                    "2. Mention job counts, role demand, skill demand, missing skills, and recommended path where available.\n"
                    "3. Do not give generic career advice.\n"
                    "4. Give a practical 30-day learning roadmap.\n"
                    "5. Keep the answer suitable for a student preparing for data jobs."
                )

                user_prompt = f"""
Job Market Context:
{market_context}

User Question:
{question}
"""

                with st.spinner("Analyzing with Groq AI & validating guardrails…"):
                    answer = ai_service.process_request(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt
                    )

                if answer.startswith("⚠️ **Security Notice**"):
                    st.warning(answer)
                else:
                    answer_placeholder = st.empty()
                    stream_answer(answer_placeholder, answer)

                with st.expander("View prompt sent to GenAI"):
                    st.code(f"System Prompt:\n{system_prompt}\n\nUser Prompt:\n{user_prompt}", language="text")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if len(st.session_state.chat_history) > 0:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()  
