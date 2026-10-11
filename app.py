import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import io
import random
import requests
from datetime import datetime

from company_search import resolve_ticker, get_company_info
from financials import (
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_price_history,
    get_current_price_data,
    format_statement_df,
    get_peers,
)
from kpi_calculations import calculate_kpis, calculate_health_score
from news import get_company_news
from ai_summary import (
    generate_executive_insights,
    generate_cfo_brief,
    chat_with_analyst,
)
from utils import (
    fmt_large, fmt_price, fmt_pct, fmt_multiple,
    build_price_chart, build_kpi_bar_chart, build_health_radar,
    build_peer_comparison_chart, build_revenue_trend_chart,
    kpi_card, health_badge, chart_config, COLORS,
    layout_defaults,
)
from private_company import parse_uploaded_file, get_available_statements
from survival_predictor import predict_survival
from export_engine import generate_pdf, generate_pptx
from variance_analysis import build_variance_table, format_variance_df
from segment_analysis import build_segment_revenue_estimates, get_segment_description
from ipo_filings import render_ipo_tracker

# ─── Load Groq API key ────────────────────────────────────────────────────────
def _load_groq_key() -> str | None:
    import os
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key and key.startswith("gsk_"):
            return key.strip()
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key.startswith("gsk_"):
        return key
    return None

# ─── Public company directory ──────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_public_company_directory() -> list[dict]:
    """Return current exchange-listed company names and tickers from SEC data."""
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    headers = {
        "User-Agent": "FinIntel AI research app (https://github.com/Hshah168/Finintel-ai)",
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            fields = payload.get("fields", [])
            rows = [
                dict(zip(fields, values))
                for values in payload["data"]
                if isinstance(values, (list, tuple))
            ]
        else:
            rows = payload.values() if isinstance(payload, dict) else payload
        companies = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).strip().upper()
            name = str(row.get("title", row.get("name", ""))).strip()
            exchange = str(row.get("exchange", "")).strip()
            if not ticker or not name or not exchange:
                continue
            key = (ticker, name)
            if key in seen:
                continue
            seen.add(key)
            companies.append({"ticker": ticker, "name": name, "exchange": exchange})
        return sorted(companies, key=lambda item: (item["name"].casefold(), item["ticker"]))
    except (requests.RequestException, ValueError, TypeError):
        return []


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinIntel AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root theme */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Hide Streamlit default elements - keep header visible for sidebar toggle */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Hide header bar background/border but keep the toggle button functional */
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* Keep sidebar collapse/expand button always visible */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: all !important;
}

/* Main background */
.stApp { background-color: #000000; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A0A0A;
    border-right: 1px solid #1C1C1E;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8E8E93 !important;
    font-size: 12px;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #1C1C1E;
    border: 1px solid #2C2C2E;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #8E8E93 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0A0A0A;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1C1C1E;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #8E8E93;
    font-weight: 500;
    font-size: 13px;
    padding: 8px 16px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1C1C1E !important;
    color: #FFFFFF !important;
}

/* Input fields */
.stTextInput > div > div > input {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0A84FF !important;
    box-shadow: 0 0 0 2px rgba(10,132,255,0.2) !important;
}

/* Buttons */
.stButton > button {
    background: #0A84FF !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #0070D8 !important;
    transform: translateY(-1px) !important;
}

/* DataFrames */
.stDataFrame {
    border: 1px solid #2C2C2E;
    border-radius: 10px;
    overflow: hidden;
}

/* Chat messages */
.stChatMessage {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E;
    border-radius: 12px;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #1C1C1E !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Section divider */
.section-divider {
    border-top: 1px solid #1C1C1E;
    margin: 24px 0;
}

/* News card */
.news-card {
    background: #1C1C1E;
    border: 1px solid #2C2C2E;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: #0A84FF55; }

/* Insight card */
.insight-card {
    background: #0A84FF0D;
    border: 1px solid #0A84FF33;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0A0A; }
::-webkit-scrollbar-thumb { background: #2C2C2E; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #48484A; }
</style>
""", unsafe_allow_html=True)

# ─── Light theme overrides ────────────────────────────────────────────────────
if st.session_state.get("theme_mode", "Dark") == "Light":
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {
        background:#F5F7FB !important; color:#111827 !important;
    }
    [data-testid="stHeader"] { background:rgba(245,247,251,.92) !important; }
    [data-testid="stSidebar"] {
        background:#FFFFFF !important; border-right:1px solid #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label { color:#4B5563 !important; }
    [data-testid="stMetric"] {
        background:#FFFFFF !important; border:1px solid #E5E7EB !important;
        border-radius:12px !important; padding:14px !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stMarkdown, .stMarkdown p { color:#111827; }
    .stTextInput input, .stTextArea textarea, [data-testid="stChatInput"] textarea {
        background:#FFFFFF !important; color:#111827 !important;
        border-color:#D1D5DB !important;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stRadio"] { color:#111827 !important; }
    .stApp [data-testid="stMarkdownContainer"] h1 { color:#111827 !important; }
    .stApp [data-testid="stMarkdownContainer"] p { color:#374151; }
    .stApp a { color:#1D4ED8 !important; }

    /* Force readable dark text across Streamlit widgets and custom components. */
    .stApp, .stApp p:not(.finintel-demo-panel *), .stApp span:not(.finintel-demo-panel *),
    .stApp label:not(.finintel-demo-panel *), .stApp li:not(.finintel-demo-panel *),
    .stApp div[data-testid="stMarkdownContainer"]:not(.finintel-demo-panel),
    .stApp div[data-testid="stMarkdownContainer"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stCaptionContainer"]:not(.finintel-demo-panel),
    .stApp [data-testid="stCaptionContainer"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stWidgetLabel"]:not(.finintel-demo-panel),
    .stApp [data-testid="stWidgetLabel"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stMetricLabel"]:not(.finintel-demo-panel *),
    .stApp [data-testid="stMetricValue"]:not(.finintel-demo-panel *),
    .stApp [data-testid="stMetricDelta"]:not(.finintel-demo-panel *),
    .stApp [data-testid="stExpander"] summary:not(.finintel-demo-panel *),
    .stApp [data-testid="stExpander"] summary *:not(.finintel-demo-panel *),
    .stApp [data-testid="stDataFrame"]:not(.finintel-demo-panel *),
    .stApp [data-testid="stTable"]:not(.finintel-demo-panel *),
    .stApp [data-testid="stAlert"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stRadio"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stCheckbox"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stSelectbox"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stMultiSelect"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stNumberInput"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stDateInput"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stFileUploader"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stTabs"] button:not(.finintel-demo-panel *),
    .stApp [data-testid="stTabs"] button *:not(.finintel-demo-panel *),
    .stApp [data-testid="stSidebar"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stHeader"] *:not(.finintel-demo-panel *),
    .stApp [data-testid="stToolbar"] *:not(.finintel-demo-panel *) {
        color:#111827 !important;
    }
    .stApp [data-testid="stMetric"] {
        background:#FFFFFF !important; border-color:#E5E7EB !important;
    }
    .stApp button[kind="primary"] *,
    .stApp button[kind="secondary"] * {
        color:#111827 !important;
    }
    .stApp input, .stApp textarea, .stApp [contenteditable="true"] {
        color:#111827 !important;
        -webkit-text-fill-color:#111827 !important;
    }
    .stApp input::placeholder, .stApp textarea::placeholder {
        color:#6B7280 !important;
        -webkit-text-fill-color:#6B7280 !important;
    }
    .stApp [data-testid="stChatInput"] textarea {
        color:#111827 !important;
        -webkit-text-fill-color:#111827 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background:#E9EDF4 !important; border-color:#D9E0EA !important;
    }
    .stTabs [aria-selected="true"] {
        background:#FFFFFF !important; color:#111827 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Session state init ────────────────────────────────────────────────────────
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "company_name" not in st.session_state:
    st.session_state.company_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "news_refresh" not in st.session_state:
    st.session_state.news_refresh = 0
if "cfo_brief" not in st.session_state:
    st.session_state.cfo_brief = None
if "upload_statements" not in st.session_state:
    st.session_state.upload_statements = None
if "upload_company_name" not in st.session_state:
    st.session_state.upload_company_name = None
if "upload_peer" not in st.session_state:
    st.session_state.upload_peer = ""
if "recent_companies" not in st.session_state:
    st.session_state.recent_companies = []  # list of (ticker, name) tuples


# ─── Unified website navigation and analysis controls ─────────────────────────
# FinIntel AI uses one full-width workspace; no separate sidebar navigation.
st.markdown("""
<style>
section[data-testid="stSidebar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] { display:none !important; visibility:hidden !important; }
.block-container { padding-top: 1.25rem !important; max-width: 1500px !important; }
.finintel-brand-row {
    display:flex; align-items:center; gap:12px; padding:5px 0 16px;
    border-bottom:1px solid #64748B33; margin-bottom:16px;
}
.finintel-brand-mark {
    width:42px;height:42px;display:flex;align-items:center;justify-content:center;
    border-radius:12px;background:linear-gradient(135deg,#2563EB,#0F766E);
    box-shadow:0 5px 14px #2563EB33;color:#FFFFFF;font-size:21px;font-weight:850;
}
.finintel-brand-name {font-size:27px;line-height:1.05;font-weight:850;color:var(--text-color,#FFFFFF);letter-spacing:-.9px;}
.finintel-brand-tag {font-size:10px;color:#8E9AAF;font-weight:700;letter-spacing:1.25px;text-transform:uppercase;margin-top:5px;}
.finintel-workspace-label {font-size:11px;font-weight:750;letter-spacing:1.3px;text-transform:uppercase;color:#60A5FA;margin:5px 0 8px;}
</style>
<div class="finintel-brand-row">
  <div class="finintel-brand-mark">F</div>
  <div><div class="finintel-brand-name">FinIntel AI</div>
  <div class="finintel-brand-tag">Financial intelligence workspace</div></div>
</div>
""", unsafe_allow_html=True)

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Company Research"
app_mode = st.radio(
    "Choose workspace",
    ["Company Research", "Private Financials"],
    horizontal=True,
    key="app_mode",
    label_visibility="collapsed",
)
st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)

if app_mode == "Company Research":
    st.markdown('<div class="finintel-workspace-label">Research any public company</div>', unsafe_allow_html=True)
    search_col, search_btn_col = st.columns([5, 1.1], gap="small")
    with search_col:
        search_input = st.text_input(
            "Company name or ticker",
            placeholder="Search a company or enter a ticker — e.g. Microsoft, AAPL, TCS…",
            label_visibility="collapsed",
            key="main_company_search",
        )
    with search_btn_col:
        search_btn = st.button("Analyze company", type="primary", use_container_width=True, key="main_analyze_company")

    quick_companies = [
        ("MSFT", "Microsoft", "Mega-cap · Technology"),
        ("WMT", "Walmart", "Mega-cap · Retail"),
        ("JPM", "JPMorgan", "Large-cap · Banking"),
        ("XOM", "ExxonMobil", "Large-cap · Energy"),
        ("LLY", "Eli Lilly", "Large-cap · Healthcare"),
        ("TM", "Toyota", "Large-cap · Automotive"),
        ("DE", "Deere", "Large-cap · Industrials"),
        ("SONY", "Sony", "Large-cap · Entertainment"),
        ("SHOP", "Shopify", "Mid-cap · Commerce"),
        ("CROX", "Crocs", "Mid-cap · Consumer"),
        ("DUOL", "Duolingo", "Mid-cap · Education tech"),
        ("SOFI", "SoFi", "Mid-cap · Fintech"),
        ("ELF", "e.l.f. Beauty", "Mid-cap · Beauty"),
        ("RKLB", "Rocket Lab", "Smaller-cap · Aerospace"),
        ("HIMS", "Hims & Hers", "Smaller-cap · Digital health"),
        ("CAVA", "CAVA", "Smaller-cap · Restaurants"),
    ]
    st.markdown('<div style="font-size:11px;color:#8E9AAF;font-weight:650;margin:5px 0 7px">QUICK LOOKUP · A MIX OF INDUSTRIES & COMPANY SIZES</div>', unsafe_allow_html=True)
    quick_cols = st.columns(8, gap="small")
    for i, (ticker, company_label, company_category) in enumerate(quick_companies):
        with quick_cols[i]:
            if st.button(company_label, key=f"quick_{ticker}", use_container_width=True, help=company_category):
                st.session_state.ticker = ticker
                st.session_state.company_name = company_label
                st.session_state.chat_history = []
                st.session_state.cfo_brief = None
                st.session_state.upload_statements = None
                st.session_state.upload_company_name = None
                st.session_state.upload_peer = ""

    public_companies = fetch_public_company_directory()
    if public_companies:
        directory_col, directory_button_col = st.columns([5, 1.1], gap="small")
        company_options = {
            f"{item['name']} ({item['ticker']})": item
            for item in public_companies
        }
        with directory_col:
            selected_company = st.selectbox(
                "Browse all listed companies",
                options=list(company_options.keys()),
                index=None,
                placeholder="Or browse the public-company directory…",
                label_visibility="collapsed",
                key="public_company_directory",
            )
        with directory_button_col:
            st.markdown('<div style="height:1px"></div>', unsafe_allow_html=True)
            analyze_selected = st.button(
                "Open selected", key="analyze_public_company",
                use_container_width=True, disabled=not selected_company,
            )
        if selected_company and analyze_selected:
            selected = company_options[selected_company]
            st.session_state.ticker = selected["ticker"]
            st.session_state.company_name = selected["name"]
            st.session_state.chat_history = []
            st.session_state.cfo_brief = None
            st.session_state.upload_statements = None
            st.session_state.upload_company_name = None
            st.session_state.upload_peer = ""

    if st.session_state.recent_companies:
        with st.expander("Recently analyzed", expanded=False):
            recent_cols = st.columns(min(5, len(st.session_state.recent_companies)), gap="small")
            for i, (recent_ticker, recent_name) in enumerate(st.session_state.recent_companies[:5]):
                with recent_cols[i]:
                    if st.button(recent_name, key=f"recent_{recent_ticker}", use_container_width=True):
                        st.session_state.ticker = recent_ticker
                        st.session_state.company_name = recent_name
                        st.session_state.chat_history = []
                        st.session_state.cfo_brief = None
                        st.rerun()
else:
    search_input = ""
    search_btn = False
    st.markdown('<div class="finintel-workspace-label">Analyze your own financial data</div>', unsafe_allow_html=True)
    upload_name_col, upload_peer_col = st.columns([1, 1], gap="medium")
    with upload_name_col:
        upload_name_input = st.text_input(
            "Company or business unit name",
            placeholder="Company, division, or cost center",
            key="upload_name",
        )
    with upload_peer_col:
        upload_peer_input = st.text_input(
            "Optional public peer ticker or name",
            placeholder="Optional benchmark peer, e.g. Microsoft",
            key="upload_peer",
        )
    upload_file_col, upload_instructions_col = st.columns([1.2, 1], gap="large")
    with upload_file_col:
        uploaded_file = st.file_uploader(
            "Upload financial statements (Excel, CSV, or PDF)",
            type=["xlsx", "xls", "csv", "pdf"],
            key="upload_file",
            help="Upload an Excel workbook, CSV, or PDF containing financial statements.",
        )
    with upload_instructions_col:
        st.markdown(
            '<div style="border:1px solid #64748B44;border-radius:12px;padding:14px 16px;'
            'font-size:12px;line-height:1.7;color:#8E9AAF">'
            '<strong style="color:#60A5FA">For best results</strong><br>'
            'Use clear row labels such as Revenue, Net Income, Total Assets, and Total Debt. '
            'Excel sheets can be named Income Statement, Balance Sheet, and Cash Flow.'
            '</div>',
            unsafe_allow_html=True,
        )
    upload_btn_col, template_col = st.columns([1.15, 5], gap="small")
    with upload_btn_col:
        upload_btn = st.button("Analyze financials", type="primary", use_container_width=True)
    with template_col:
        sample = {
            "Income Statement": pd.DataFrame({
                "Line Item": ["Total Revenue","Gross Profit","Operating Income","Net Income","EBITDA"],
                "2022": [50e6,20e6,8e6,5e6,10e6],
                "2023": [60e6,25e6,10e6,7e6,13e6],
                "2024": [72e6,31e6,13e6,9e6,16e6],
            }),
            "Balance Sheet": pd.DataFrame({
                "Line Item": ["Total Current Assets","Total Assets","Total Current Liabilities","Total Debt","Total Stockholder Equity"],
                "2022": [15e6,45e6,8e6,12e6,25e6],
                "2023": [18e6,52e6,9e6,10e6,30e6],
                "2024": [22e6,61e6,10e6,8e6,37e6],
            }),
            "Cash Flow": pd.DataFrame({
                "Line Item": ["Operating Cash Flow","Capital Expenditure","Free Cash Flow"],
                "2022": [8e6,-2e6,6e6],
                "2023": [11e6,-3e6,8e6],
                "2024": [14e6,-3.5e6,10.5e6],
            }),
        }
        template_buf = io.BytesIO()
        with pd.ExcelWriter(template_buf, engine="openpyxl") as writer:
            for sheet_name, df in sample.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        template_buf.seek(0)
        st.download_button(
            "Download Excel template",
            data=template_buf.getvalue(),
            file_name="FinIntel_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if upload_btn:
        if uploaded_file:
            with st.spinner("Parsing your financials…"):
                stmts, detected = parse_uploaded_file(uploaded_file)
                upload_company = upload_name_input.strip() or detected or "My Company"
                available_statements = get_available_statements(stmts)
            if available_statements:
                st.session_state.upload_statements = stmts
                st.session_state.upload_company_name = upload_company
                st.session_state.upload_peer = upload_peer_input.strip()
                st.session_state.ticker = None
                st.success(f"Parsed: {', '.join(available_statements)}")
                st.rerun()
            else:
                st.error("Could not extract financial statements. Check row labels and year columns.")
        else:
            st.warning("Please upload a financial file first.")

groq_key = _load_groq_key()

# ─── Social links in the website header ───────────────────────────────────────
header_spacer, linkedin_col, github_col = st.columns([10, 0.65, 0.65], gap="small")
with linkedin_col:
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;padding-top:5px">'
        '<a aria-label="LinkedIn profile" title="LinkedIn profile" '
        'href="https://www.linkedin.com/in/shah-hetal/" target="_blank" rel="noopener noreferrer" '
        'style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;'
        'border:1px solid #64748B55;border-radius:10px;color:#0A66C2;text-decoration:none">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" '
        'fill="currentColor" aria-hidden="true"><path d="M19 3A2 2 0 0 1 21 5V19A2 2 0 0 1 19 21H5A2 2 0 0 1 3 19V5A2 2 0 0 1 5 3H19ZM8.34 17.34V10H5.67V17.34H8.34ZM7 8.99A1.55 1.55 0 1 0 7 5.89A1.55 1.55 0 0 0 7 8.99ZM18.34 17.34V13.32C18.34 11.17 17.19 10.17 15.66 10.17C14.42 10.17 13.86 10.85 13.55 11.33V10H10.88V17.34H13.55V13.75C13.55 12.8 13.73 11.88 14.91 11.88C16.08 11.88 16.1 12.97 16.1 13.81V17.34H18.34Z"/></svg>'
        '</a></div>',
        unsafe_allow_html=True,
    )
with github_col:
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;padding-top:5px">'
        '<a aria-label="GitHub profile" title="GitHub profile" '
        'href="https://github.com/Hshah168" target="_blank" rel="noopener noreferrer" '
        'style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;'
        'border:1px solid #64748B55;border-radius:10px;color:var(--text-color,#FFFFFF);text-decoration:none">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" '
        'fill="currentColor" aria-hidden="true"><path d="M12 .9A11.1 11.1 0 0 0 8.49 22.53c.55.1.76-.24.76-.53v-2.08c-3.1.68-3.76-1.32-3.76-1.32-.5-1.29-1.24-1.63-1.24-1.63-1.02-.7.08-.69.08-.69 1.13.08 1.73 1.16 1.73 1.16 1 1.72 2.63 1.22 3.27.93.1-.72.39-1.22.71-1.5-2.48-.28-5.09-1.24-5.09-5.53 0-1.22.44-2.22 1.16-3-.12-.28-.5-1.42.11-2.96 0 0 .95-.3 3.05 1.15a10.6 10.6 0 0 1 5.55 0c2.1-1.45 3.05-1.15 3.05-1.15.61 1.54.23 2.68.11 2.96.72.78 1.16 1.78 1.16 3 0 4.3-2.61 5.25-5.1 5.52.4.35.75 1.03.75 2.08V22c0 .29.2.63.76.52A11.1 11.1 0 0 0 12 .9Z"/></svg>'
        '</a></div>',
        unsafe_allow_html=True,
    )

# ─── Load FMP key for IPO tracker ─────────────────────────────────────────────
def _load_fmp_key() -> str:
    try:
        k = st.secrets.get("FMP_API_KEY", "")
        if k and k != "your_fmp_key_here": return k
    except Exception: pass
    import os
    return os.environ.get("FMP_API_KEY", "")


# ─── Search handler ────────────────────────────────────────────────────────────
if search_btn and search_input.strip():
    with st.spinner(f"Identifying {search_input}..."):
        ticker, full_name = resolve_ticker(search_input.strip())
    if ticker == "PRIVATE":
        st.error(f"**{search_input}** is a privately held company. Switch to Upload Mode to analyze private financials.")
    elif ticker:
        st.session_state.ticker = ticker
        st.session_state.company_name = full_name
        st.session_state.chat_history = []
        st.session_state.cfo_brief = None
        st.session_state.upload_statements = None
        st.session_state.upload_company_name = None
        st.session_state.upload_peer = ""
    else:
        st.error(f"Could not identify a publicly traded company for **{search_input}**. Try a ticker directly.")


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD MODE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Private Financials":
    stmts = st.session_state.upload_statements
    uname = st.session_state.upload_company_name or "My Company"
    upeer = st.session_state.upload_peer or ""

    if stmts is None:
        avail = []
    else:
        avail = get_available_statements(stmts)

    if stmts and avail:
        u_kpis = calculate_kpis(stmts["income"], stmts["balance"], stmts["cashflow"], {})
        u_score, u_label, u_breakdown = calculate_health_score(u_kpis, {})
    else:
        u_kpis = {}
        u_score, u_label, u_breakdown = 0, "N/A", {
            "Profitability": 0,
            "Growth": 0,
            "Liquidity": 0,
            "Leverage": 0,
            "Cash Flow": 0,
        }

    if stmts is None or not avail:

        st.markdown(
            """
            <p style="
                font-size:16px;
                color:#8E8E93;
                margin:0 0 12px;
                max-width:500px;
            ">
            Analyze any company's internal financials. Public or private,<br>
            listed or unlisted, if you have the numbers, we can analyze them.
            </p>
            """,
            unsafe_allow_html=True
        )


        use_cols = st.columns(3)

        use_cases = [
            (
                "Internal Business Units",
                "Analyze divisions, cost centers, management accounts, and internal P&Ls."
            ),
            (
                "Private Companies",
                "Generate KPIs, Health Scores, and CFO Briefs from your own financial statements."
            ),
            (
                "Any Financial Dataset",
                "Works with startups, nonprofits, subsidiaries, joint ventures, or any organization with financial statements."
            ),
        ]


        for col, (title, desc) in zip(use_cols, use_cases):

            with col:

                st.markdown(
                    f"""
                    <div style="
                        background-color:#1C1C1E;
                        border:1px solid #2C2C2E;
                        border-radius:14px;
                        padding:20px;
                        height:140px;
                        text-align:center;
                    ">

                    <h4 style="
                        color:#FFFFFF;
                        font-size:14px;
                        font-weight:700;
                        margin:0 0 12px 0;
                    ">
                    {title}
                    </h4>


                    <p style="
                        color:#8E8E93;
                        font-size:12px;
                        line-height:1.5;
                        margin:0;
                    ">
                    {desc}
                    </p>


                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.markdown(
            """
            <p style="
                text-align:center;
                color:#48484A;
                font-size:13px;
                margin-top:24px;
            ">
            Use the upload controls above and click <b>Analyze financials</b> to begin.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.stop()

    # KPI strip
    strip_data = [
        ("Revenue Growth", u_kpis.get("Revenue Growth %",(None,"N/A"))[1]),
        ("Gross Margin",   u_kpis.get("Gross Margin %",(None,"N/A"))[1]),
        ("Net Margin",     u_kpis.get("Net Margin %",(None,"N/A"))[1]),
        ("Current Ratio",  u_kpis.get("Current Ratio",(None,"N/A"))[1]),
        ("Debt-to-Equity", u_kpis.get("Debt-to-Equity",(None,"N/A"))[1]),
    ]
    for col, (label, val) in zip(st.columns(5), strip_data):
        col.metric(label, val)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Tabs for upload mode
    u_tabs = st.tabs(["Financials", "KPIs & Health", "Insights", "CFO Brief", "Peer Compare"])

    # ── U-Tab 1: Financials ───────────────────────────────────────────────────
    with u_tabs[0]:
        st.markdown("**Uploaded Financial Statements**")
        stmt_map = {"Income Statement": "income", "Balance Sheet": "balance", "Cash Flow": "cashflow"}
        avail_stmts = [k for k, v in stmt_map.items() if not stmts[v].empty]
        if avail_stmts:
            for tab, label in zip(st.tabs(avail_stmts), avail_stmts):
                with tab:
                    df = stmts[stmt_map[label]]
                    def fmt_u(x):
                        try:
                            v = float(x)
                            if abs(v) >= 1e9: return f"${v/1e9:,.2f}B"
                            if abs(v) >= 1e6: return f"${v/1e6:,.1f}M"
                            if abs(v) >= 1e3: return f"${v/1e3:,.0f}K"
                            return f"${v:,.0f}"
                        except: return "-"
                    try:
                        display_df = df.map(fmt_u)
                    except AttributeError:
                        display_df = df.applymap(fmt_u)
                    st.dataframe(display_df, use_container_width=True)
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf)
                    st.download_button(f"Download {label} CSV", csv_buf.getvalue(),
                                       f"{uname}_{label}.csv", "text/csv", key=f"u_dl_{label}")

    # ── U-Tab 2: KPIs & Health ────────────────────────────────────────────────
    with u_tabs[1]:
        sc, rc = st.columns([1,1])
        with sc:
            st.markdown("**Financial Health Score**")
            st.markdown(health_badge(u_label, u_score), unsafe_allow_html=True)
            st.markdown("**Score Breakdown**")
            for dim, score in u_breakdown.items():
                pct = score / 20
                color = (COLORS["success"] if pct>=0.75 else COLORS["primary"] if pct>=0.50
                         else COLORS["warning"] if pct>=0.25 else COLORS["danger"])
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="color:#FFFFFF;font-size:13px">{dim}</span>
                        <span style="color:{color};font-size:13px;font-weight:600">{score}/20</span>
                    </div>
                    <div style="background:#2C2C2E;border-radius:4px;height:6px">
                        <div style="background:{color};border-radius:4px;height:6px;width:{pct*100:.0f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with rc:
            st.markdown("**Dimension Radar**")
            st.plotly_chart(build_health_radar(u_breakdown), use_container_width=True, config=chart_config())

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Key Performance Indicators**")
        kpi_groups = {
            "Profitability": ["Revenue Growth %","Gross Margin %","Operating Margin %","Net Margin %","EBITDA Margin %"],
            "Efficiency": ["ROA %","ROE %","FCF Margin %"],
            "Liquidity & Leverage": ["Current Ratio","Quick Ratio","Debt-to-Equity"],
        }
        for group_name, kpi_names in kpi_groups.items():
            st.markdown(f"**{group_name}**")
            cols = st.columns(len(kpi_names))
            for col, kn in zip(cols, kpi_names):
                _, fmt_str, delta = u_kpis.get(kn, (None,"N/A",None))
                col.metric(kn, fmt_str)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Margin Profile**")
        mc = build_kpi_bar_chart(u_kpis)
        if mc.data:
            st.plotly_chart(mc, use_container_width=True, config=chart_config())

    # ── U-Tab 3: Insights ─────────────────────────────────────────────────────
    with u_tabs[2]:
        st.markdown("**Executive Insights**")
        with st.spinner("Generating insights..."):
            u_insights = generate_executive_insights(
                uname, "UPLOAD", u_kpis, u_score, u_label, {},
                api_key=groq_key or None,
            )
        icons = {"Revenue Trend":"","Profitability Trend":"","Balance Sheet Strength":"","Cash Flow Analysis":""}
        for title, text in u_insights.items():
            st.markdown(f"""
            <div class="insight-card">
                <p style="color:#0A84FF;font-size:11px;text-transform:uppercase;
                          letter-spacing:0.8px;font-weight:700;margin:0 0 6px">{title}</p>
                <p style="color:#FFFFFF;font-size:14px;line-height:1.6;margin:0">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── U-Tab 4: CFO Brief ────────────────────────────────────────────────────
    with u_tabs[3]:
        st.markdown("**CFO Brief Generator**")
        st.markdown('<p style="color:#8E8E93;font-size:13px">Generate a structured executive brief from your uploaded financials.</p>', unsafe_allow_html=True)
        if st.button("Generate CFO Brief", key="u_cfo_btn"):
            with st.spinner("Compiling CFO Brief..."):
                u_brief = generate_cfo_brief(
                    uname, "UPLOAD", {}, u_kpis, u_score, u_label, [],
                    api_key=groq_key or None,
                )
            st.markdown(f'<div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:24px">{u_brief}</div>', unsafe_allow_html=True)
            st.download_button("Download CFO Brief", u_brief,
                               f"{uname}_CFO_Brief_{datetime.now().strftime('%Y%m%d')}.md",
                               "text/markdown", key="u_dl_brief")

    # ── U-Tab 5: Peer Compare ─────────────────────────────────────────────────
    with u_tabs[4]:
        st.markdown("**Peer Comparison**")
        peer_query = upeer or st.text_input("Enter a public company to benchmark against",
                                             placeholder="Microsoft, SAP, Apple...",
                                             key="u_peer_inline")
        if peer_query:
            with st.spinner(f"Loading {peer_query} data..."):
                pt, pname = resolve_ticker(peer_query)
            if pt and pt != "PRIVATE":
                p_info  = get_company_info(pt)
                p_kpis  = calculate_kpis(get_income_statement(pt), get_balance_sheet(pt), get_cash_flow(pt), p_info)
                p_score, p_label, _ = calculate_health_score(p_kpis, p_info)

                # Table
                compare_metrics = ["Revenue Growth %","Gross Margin %","Net Margin %",
                                   "Operating Margin %","ROE %","Current Ratio","Debt-to-Equity","FCF Margin %"]
                rows = [{"Metric": m,
                         uname: u_kpis.get(m,(None,"N/A"))[1],
                         pname: p_kpis.get(m,(None,"N/A"))[1]}
                        for m in compare_metrics]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Bar chart
                margin_metrics = ["Gross Margin %","Operating Margin %","Net Margin %","FCF Margin %"]
                labels, uvals, pvals = [], [], []
                for m in margin_metrics:
                    uv = u_kpis.get(m,(None,))[0]
                    pv = p_kpis.get(m,(None,))[0]
                    if uv is not None and pv is not None:
                        labels.append(m.replace(" %",""))
                        uvals.append(uv)
                        pvals.append(pv)
                if labels:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name=uname, x=labels, y=uvals,
                                         marker_color=COLORS["primary"],
                                         text=[f"{v:.1f}%" for v in uvals], textposition="outside"))
                    fig.add_trace(go.Bar(name=pname, x=labels, y=pvals,
                                         marker_color=COLORS["neutral"],
                                         text=[f"{v:.1f}%" for v in pvals], textposition="outside"))
                    layout = layout_defaults("Margin Comparison", height=350)
                    layout["barmode"] = "group"
                    layout["yaxis"]["ticksuffix"] = "%"
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True, config=chart_config())

                # Health score cards
                h1, h2 = st.columns(2)
                for col, name, sc, lb in [(h1,uname,u_score,u_label),(h2,pname,p_score,p_label)]:
                    hc = (COLORS["success"] if sc>=75 else COLORS["primary"] if sc>=55
                          else COLORS["warning"] if sc>=35 else COLORS["danger"])
                    with col:
                        st.markdown(f"""
                        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;
                                    padding:20px;text-align:center">
                            <p style="color:#8E8E93;font-size:12px;margin:0 0 8px">{name}</p>
                            <p style="font-size:40px;font-weight:800;color:{hc};margin:0">{sc}</p>
                            <p style="color:{hc};font-size:14px;font-weight:600;margin:4px 0 0">{lb}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"Could not find '{peer_query}'. Try a ticker like MSFT or AAPL.")
        else:
            st.info("Enter a public company name above to benchmark your financials against it.")

    st.stop()  # Don't render search mode dashboard in upload mode


# ─── Landing page (no company selected) ───────────────────────────────────────
DEMO_COMPANIES = [
    {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology"},
    {"ticker": "AAPL", "name": "Apple", "sector": "Consumer Technology"},
    {"ticker": "NVDA", "name": "NVIDIA", "sector": "Semiconductors"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financial Services"},
    {"ticker": "WMT", "name": "Walmart", "sector": "Retail"},
    {"ticker": "XOM", "name": "ExxonMobil", "sector": "Energy"},
    {"ticker": "LLY", "name": "Eli Lilly", "sector": "Healthcare"},
    {"ticker": "TM", "name": "Toyota", "sector": "Automotive"},
    {"ticker": "SHOP", "name": "Shopify", "sector": "Commerce Software"},
    {"ticker": "CROX", "name": "Crocs", "sector": "Consumer Products"},
]

if not st.session_state.ticker:
    # Pick one real public company per visitor session; financial data is shared
    # through the existing one-hour Streamlit caches.
    if "landing_demo_ticker" not in st.session_state:
        st.session_state.landing_demo_ticker = random.choice(DEMO_COMPANIES)["ticker"]
    demo = next(
        (item for item in DEMO_COMPANIES
         if item["ticker"] == st.session_state.landing_demo_ticker),
        DEMO_COMPANIES[0],
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _landing_company_snapshot(ticker: str, company_name: str) -> dict:
        info = get_company_info(ticker)
        income = get_income_statement(ticker)
        balance = get_balance_sheet(ticker)
        cashflow = get_cash_flow(ticker)
        kpis = calculate_kpis(income, balance, cashflow, info)
        score, label, _ = calculate_health_score(kpis, info)
        brief = generate_cfo_brief(
            info.get("longName") or info.get("shortName") or company_name,
            ticker, info, kpis, score, label, [], api_key=None,
        )
        summary = ""
        if "## Executive Summary" in brief:
            summary = brief.split("## Executive Summary", 1)[1].split("\n## ", 1)[0].strip()
        return {
            "info": info, "kpis": kpis, "score": score, "label": label,
            "summary": summary, "income": income,
        }

    # Warm the three featured demo companies into Streamlit's shared cache so
    # selecting one usually reuses the financial statements and company profile.
    if "featured_demo_cache_warmed" not in st.session_state:
        for featured_ticker, featured_name in [
            ("MSFT", "Microsoft"), ("AAPL", "Apple"), ("NVDA", "NVIDIA")
        ]:
            try:
                _landing_company_snapshot(featured_ticker, featured_name)
            except Exception:
                pass
        st.session_state.featured_demo_cache_warmed = True

    try:
        snapshot = _landing_company_snapshot(demo["ticker"], demo["name"])
    except Exception:
        snapshot = {"info": {}, "kpis": {}, "score": None, "label": "Unavailable",
                    "summary": "", "income": pd.DataFrame()}

    info = snapshot.get("info", {})
    kpis = snapshot.get("kpis", {})
    score = snapshot.get("score")
    score_label = snapshot.get("label", "Unavailable")
    score_color = "#34D399" if isinstance(score, (int, float)) and score >= 70 else (
        "#FBBF24" if isinstance(score, (int, float)) and score >= 45 else "#F87171"
    )

    def _demo_kpi(name: str) -> str:
        item = kpis.get(name)
        return item[1] if item and len(item) > 1 else "N/A"

    rev_growth = _demo_kpi("Revenue Growth %")
    gross_margin = _demo_kpi("Gross Margin %")
    net_margin = _demo_kpi("Net Margin %")
    fiscal_period = "Latest reported annual financials"
    try:
        income_df = snapshot.get("income")
        if income_df is not None and not income_df.empty:
            period = income_df.columns[0]
            fiscal_period = period.strftime("%Y fiscal year") if hasattr(period, "strftime") else str(period)
    except Exception:
        pass

    # Two-part product-first landing page: concise positioning + live company snapshot.
    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        st.markdown(
            '<div style="padding:42px 8px 18px 0">'
            '<div style="color:#60A5FA;font-size:11px;font-weight:700;'
            'letter-spacing:1.7px;text-transform:uppercase;margin-bottom:18px">'
            'FINANCIAL INTELLIGENCE, IN ACTION</div>'
            '<h1 style="font-size:48px;line-height:1.04;letter-spacing:-2px;'
            'font-weight:800;color:var(--landing-heading,#FFFFFF);margin:0 0 26px">'
            'Know the numbers.<br>Understand the business.</h1>'
            '<p style="font-size:17px;line-height:1.6;color:var(--landing-copy,#B8C1CF);'
            'margin:0 0 16px">Financial intelligence that turns company statements into '
            'clear performance signals, peer context, and executive-ready insights.</p>'
            '<p style="font-size:15px;line-height:1.6;color:var(--landing-copy,#B8C1CF);'
            'margin:0 0 16px">Built for FP&amp;A professionals, financial analysts, investors, '
            'and business leaders who need to understand what is driving performance.</p>'
            '<p style="font-size:14px;line-height:1.6;color:var(--landing-muted,#8E9AAF);'
            'margin:0">The 24-month Survival Predictor flags potential financial distress signals '
            'to help teams investigate risk earlier-not as a guarantee or investment recommendation.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="finintel-demo-panel" style="background:linear-gradient(145deg,#111C2E,#0D1421);'
            f'border:1px solid #293A53;border-radius:20px;padding:23px 24px 20px;'
            f'margin-top:24px;box-shadow:0 18px 50px rgba(0,0,0,.22)">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'
            f'margin-bottom:18px">'
            f'<div><div style="color:#8FA4C1;font-size:10px;font-weight:700;letter-spacing:1.5px;'
            f'margin-bottom:7px">LIVE COMPANY SNAPSHOT</div>'
            f'<div style="font-size:23px;font-weight:750;color:#FFFFFF;line-height:1.2">'
            f'{demo["name"]} <span style="font-size:13px;color:#8FA4C1">({demo["ticker"]})</span></div>'
            f'<div style="font-size:12px;color:#9CAEC4;margin-top:5px">'
            f'{info.get("sector") or demo["sector"]} · {fiscal_period}</div></div>'
            f'<div style="background:#102D2A;border:1px solid #1D5C4D;border-radius:999px;'
            f'padding:6px 10px;color:#6EE7B7;font-size:10px;font-weight:700;white-space:nowrap">'
            f'PUBLIC FILINGS</div></div>'
            f'<div style="background:#131F30;border:1px solid #283A52;border-radius:14px;'
            f'padding:17px;margin-bottom:12px;display:flex;align-items:center;gap:18px">'
            f'<div style="flex:1"><div style="color:#93A4BA;font-size:11px;margin-bottom:5px">'
            f'Financial Health Score</div><div style="color:{score_color};font-size:38px;'
            f'font-weight:800;line-height:1.1">{score if score is not None else "N/A"}'
            f'<span style="font-size:13px;color:#91A1B7;font-weight:500"> / 100</span></div>'
            f'<div style="color:{score_color};font-size:12px;font-weight:600;margin-top:5px">'
            f'{score_label}</div></div>'
            f'<div style="width:1px;height:62px;background:#2A3A50"></div>'
            f'<div style="flex:1"><div style="color:#93A4BA;font-size:11px;margin-bottom:7px">'
            f'Revenue growth</div><div style="color:#FFFFFF;font-size:25px;font-weight:750">'
            f'{rev_growth}</div><div style="color:#7F91A8;font-size:10px;margin-top:4px">'
            f'Year over year</div></div></div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:15px">'
            f'<div style="background:#131F30;border:1px solid #283A52;border-radius:12px;padding:13px">'
            f'<div style="font-size:10px;color:#93A4BA;margin-bottom:7px">GROSS MARGIN</div>'
            f'<div style="font-size:23px;color:#FFFFFF;font-weight:750">{gross_margin}</div></div>'
            f'<div style="background:#131F30;border:1px solid #283A52;border-radius:12px;padding:13px">'
            f'<div style="font-size:10px;color:#93A4BA;margin-bottom:7px">NET MARGIN</div>'
            f'<div style="font-size:23px;color:#FFFFFF;font-weight:750">{net_margin}</div></div></div>'
            f'<div style="border-top:1px solid #293A53;padding-top:14px">'
            f'<div style="font-size:10px;color:#60A5FA;font-weight:700;letter-spacing:1.3px;'
            f'margin-bottom:7px">CFO BRIEF · FINANCIAL SIGNAL</div>'
            f'<p style="font-size:12px;line-height:1.65;color:#C4CEDD;margin:0">'
            f'{snapshot.get("summary") or "Financial data is loading or not available for this company. Choose a company below to open its full analysis."}</p></div>'
            f'<div style="font-size:10px;color:#75869D;margin-top:13px">'
            f'Source: Yahoo Finance · {fiscal_period} · Cached for faster repeat visits</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Company search follows the intro snapshot so visitors first understand the product.
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="border-top:1px solid #263449;padding-top:22px">'
        '<p style="font-size:11px;font-weight:700;letter-spacing:1.4px;color:#60A5FA;'
        'text-transform:uppercase;margin:0 0 7px">Start your research</p>'
        '<h2 style="font-size:24px;font-weight:750;letter-spacing:-.5px;'
        'color:#FFFFFF;margin:0 0 5px">Which company do you want to understand?</h2>'
        '<p style="font-size:13px;color:#9CAEC4;margin:0 0 14px">Search a ticker or choose from companies across industries and sizes.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    landing_search_col, landing_search_btn_col = st.columns([5, 1.15], gap="small")
    with landing_search_col:
        landing_search_input = st.text_input(
            "Company name or ticker",
            placeholder="Search any company — e.g. Microsoft, SOFI, Toyota, CAVA…",
            label_visibility="collapsed",
            key="landing_company_search",
        )
    with landing_search_btn_col:
        landing_search_btn = st.button("Analyze company", type="primary", use_container_width=True, key="landing_analyze_company")
    if landing_search_btn and landing_search_input.strip():
        with st.spinner(f"Identifying {landing_search_input.strip()}…"):
            landing_ticker, landing_name = resolve_ticker(landing_search_input.strip())
        if landing_ticker == "PRIVATE":
            st.warning("This appears to be a private company. Switch to Private Financials to upload its statements.")
        elif landing_ticker:
            st.session_state.ticker = landing_ticker
            st.session_state.company_name = landing_name
            st.session_state.chat_history = []
            st.session_state.cfo_brief = None
            st.session_state.upload_statements = None
            st.session_state.upload_company_name = None
            st.session_state.upload_peer = ""
            st.rerun()
        else:
            st.error("Could not identify that company. Try its stock ticker.")

    # ── Homepage financial charts ─────────────────────────────────────────────
    st.markdown('<div style="height:34px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="border-top:1px solid #263449;padding-top:25px;margin-bottom:4px">'
        '<p style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#60A5FA;'
        'text-transform:uppercase;margin:0 0 8px">A closer look at performance</p>'
        '<h2 style="font-size:26px;font-weight:750;letter-spacing:-.7px;'
        'color:var(--landing-heading,#FFFFFF);margin:0 0 7px">See the story behind the numbers.</h2>'
        '<p style="font-size:13px;line-height:1.6;color:var(--landing-copy,#9CAEC4);'
        'margin:0 0 18px">Explore reported financial trends, then compare the business with its peers.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns([1.3, 0.9], gap="large")
    with chart_left:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;margin:0 0 4px">Revenue & operating income</div>'
            '<div style="font-size:11px;color:#8998AD;margin-bottom:8px">Annual reported financials · USD billions</div>',
            unsafe_allow_html=True,
        )
        chart_income = snapshot.get("income")
        revenue_key = next(
            (key for key in ["Total Revenue", "Revenue", "Net Revenue", "Revenues"]
             if chart_income is not None and not chart_income.empty and key in chart_income.index),
            None,
        )
        operating_key = next(
            (key for key in ["Operating Income", "Operating Income Loss"]
             if chart_income is not None and not chart_income.empty and key in chart_income.index),
            None,
        )
        if chart_income is not None and not chart_income.empty and revenue_key:
            trend_fig = go.Figure()
            fiscal_cols = list(chart_income.columns)[::-1]
            fiscal_years = [
                col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]
                for col in fiscal_cols
            ]
            revenue_values = [
                float(chart_income.loc[revenue_key, col]) / 1e9
                if pd.notna(chart_income.loc[revenue_key, col]) else None
                for col in fiscal_cols
            ]
            trend_fig.add_trace(go.Scatter(
                x=fiscal_years, y=revenue_values, name="Revenue",
                mode="lines+markers", line=dict(color="#60A5FA", width=3),
                marker=dict(size=7), connectgaps=False,
                hovertemplate="FY %{x}<br>Revenue: $%{y:.2f}B<extra></extra>",
            ))
            if operating_key:
                operating_values = [
                    float(chart_income.loc[operating_key, col]) / 1e9
                    if pd.notna(chart_income.loc[operating_key, col]) else None
                    for col in fiscal_cols
                ]
                trend_fig.add_trace(go.Scatter(
                    x=fiscal_years, y=operating_values, name="Operating income",
                    mode="lines+markers", line=dict(color="#34D399", width=2.5),
                    marker=dict(size=6), connectgaps=False,
                    hovertemplate="FY %{x}<br>Operating income: $%{y:.2f}B<extra></extra>",
                ))
            trend_layout = layout_defaults("", height=330)
            trend_layout["margin"] = dict(l=18, r=18, t=20, b=28)
            trend_layout["yaxis"]["tickprefix"] = "$"
            trend_layout["yaxis"]["ticksuffix"] = "B"
            chart_text = "#FFFFFF"
            chart_muted = "#8E8E93"
            chart_grid = "#2C2C2E"
            trend_layout["font"] = dict(color=chart_text, family="Inter, system-ui, sans-serif")
            trend_layout["xaxis"].update(gridcolor=chart_grid, linecolor=chart_grid, tickfont=dict(color=chart_muted))
            trend_layout["yaxis"].update(gridcolor=chart_grid, linecolor=chart_grid, tickfont=dict(color=chart_muted))
            trend_layout["legend"] = dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_text),
            )
            trend_fig.update_layout(**trend_layout)
            st.plotly_chart(
                trend_fig, use_container_width=True,
                config={"displayModeBar": False, "displaylogo": False},
                key=f"landing_financial_trend_{demo['ticker']}",
            )
        else:
            st.info("Annual revenue history is not available for this sample company right now.")

    with chart_right:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;margin:0 0 4px">Peer benchmark</div>'
            '<div style="font-size:11px;color:#8998AD;line-height:1.6;margin-bottom:12px">'
            'Compare profitability and growth with suggested peers.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:#101C2D;border:1px solid #26384F;border-radius:13px;'
            'padding:15px 16px;margin-bottom:12px">'
            '<div style="font-size:10px;color:#93A4BA;margin-bottom:8px">COMPANY IN FOCUS</div>'
            f'<div style="font-size:19px;font-weight:750;color:#FFFFFF">{demo["name"]}</div>'
            f'<div style="font-size:12px;color:#93A4BA;margin-top:4px">{demo["ticker"]} · {demo["sector"]}</div>'
            '<div style="font-size:12px;color:#C4CEDD;line-height:1.6;margin-top:10px">'
            'Benchmark revenue growth and margins to see where performance stands out.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        show_peer_chart = st.checkbox(
            "Load live peer comparison",
            key=f"landing_load_peers_{demo['ticker']}",
            help="Loads available financial statements for this company and its suggested peers.",
        )
        if show_peer_chart:
            with st.spinner("Loading peer financials..."):
                peer_tickers = get_peers(demo["ticker"], info)[:3]
                peer_rows = []
                for peer_ticker in [demo["ticker"]] + peer_tickers:
                    try:
                        if peer_ticker == demo["ticker"]:
                            peer_info = info
                            peer_kpis = kpis
                        else:
                            peer_info = get_company_info(peer_ticker)
                            peer_kpis = calculate_kpis(
                                get_income_statement(peer_ticker),
                                get_balance_sheet(peer_ticker),
                                get_cash_flow(peer_ticker),
                                peer_info,
                            )
                        peer_name = peer_info.get("shortName") or peer_info.get("longName") or peer_ticker
                        for metric_name in ["Revenue Growth %", "Operating Margin %", "Net Margin %"]:
                            metric_item = peer_kpis.get(metric_name)
                            metric_value = metric_item[0] if metric_item else None
                            if metric_value is not None and pd.notna(metric_value):
                                peer_rows.append({
                                    "Company": peer_name,
                                    "Metric": metric_name.replace(" %", ""),
                                    "Value": float(metric_value),
                                })
                    except Exception:
                        continue
            if peer_rows:
                peer_fig = go.Figure()
                palette = ["#60A5FA", "#34D399", "#FBBF24", "#A78BFA"]
                company_labels = list(dict.fromkeys(row["Company"] for row in peer_rows))
                for index, company_label in enumerate(company_labels):
                    selected_rows = [row for row in peer_rows if row["Company"] == company_label]
                    peer_fig.add_trace(go.Bar(
                        name=company_label,
                        x=[row["Metric"] for row in selected_rows],
                        y=[row["Value"] for row in selected_rows],
                        marker_color=palette[index % len(palette)],
                        hovertemplate="%{x}: %{y:.1f}%<extra>" + company_label + "</extra>",
                    ))
                peer_layout = layout_defaults("", height=330)
                peer_layout["barmode"] = "group"
                peer_layout["margin"] = dict(l=12, r=12, t=24, b=36)
                peer_layout["yaxis"]["ticksuffix"] = "%"
                peer_layout["font"] = dict(color=chart_text, family="Inter, system-ui, sans-serif")
                peer_layout["xaxis"].update(gridcolor=chart_grid, linecolor=chart_grid, tickfont=dict(color=chart_muted))
                peer_layout["yaxis"].update(gridcolor=chart_grid, linecolor=chart_grid, tickfont=dict(color=chart_muted))
                peer_layout["legend"] = dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=chart_text),
                )
                peer_fig.update_layout(**peer_layout)
                st.plotly_chart(
                    peer_fig, use_container_width=True,
                    config={"displayModeBar": False, "displaylogo": False},
                    key=f"landing_peer_chart_{demo['ticker']}",
                )
                st.caption("Metrics use available reported financials; fiscal periods and business models may differ.")
            else:
                st.info("Comparable financial metrics are not available for these peers right now.")
        else:
            st.markdown(
                '<div style="border:1px dashed #35465D;border-radius:10px;padding:13px 14px;'
                'color:#93A4BA;font-size:12px;line-height:1.6">'
                'Load an on-demand chart comparing revenue growth, operating margin, and net margin.'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11px;font-weight:700;letter-spacing:1.4px;'
        'color:#8998AD;text-transform:uppercase;margin:0 0 10px">Explore a live analysis</p>',
        unsafe_allow_html=True,
    )
    try_cols = st.columns(3, gap="small")
    for col, (try_ticker, try_name) in zip(
        try_cols, [("MSFT", "Microsoft"), ("AAPL", "Apple"), ("NVDA", "NVIDIA")]
    ):
        with col:
            if st.button(f"Try {try_name}  ↗", key=f"landing_try_{try_ticker}",
                         use_container_width=True):
                st.session_state.ticker = try_ticker
                st.session_state.company_name = try_name
                st.session_state.chat_history = []
                st.session_state.cfo_brief = None
                st.session_state.upload_statements = None
                st.session_state.upload_company_name = None
                st.session_state.upload_peer = ""
                st.rerun()



    # IPO discovery is intentionally placed at the bottom, after the product-first
    # landing experience and company demo.
    st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
    render_ipo_tracker(compact=True)

    st.stop()


# ─── Main dashboard ────────────────────────────────────────────────────────────
ticker = st.session_state.ticker
company_display_name = st.session_state.company_name or ticker

# Load all data
with st.spinner(f"Loading {company_display_name} data..."):
    info = get_company_info(ticker)
    price_data = get_current_price_data(ticker)
    income = get_income_statement(ticker)
    balance = get_balance_sheet(ticker)
    cashflow = get_cash_flow(ticker)
    kpis = calculate_kpis(income, balance, cashflow, info)
    health_score, health_label, health_breakdown = calculate_health_score(kpis, info)

full_name = info.get("longName") or info.get("shortName") or company_display_name
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")
country = info.get("country", "N/A")
website = info.get("website", "")
employees = info.get("fullTimeEmployees")
summary = info.get("longBusinessSummary", "No business description available.")

price = price_data.get("price", 0) or 0
change = price_data.get("change", 0) or 0
change_pct = price_data.get("change_pct", 0) or 0

# ─── Company header ────────────────────────────────────────────────────────────
change_color = COLORS["success"] if change >= 0 else COLORS["danger"]
change_arrow = "▲" if change >= 0 else "▼"
market_cap = info.get("marketCap", 0)
enterprise_val = info.get("enterpriseValue", 0)

st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;
    padding:20px 0 16px;border-bottom:1px solid #1C1C1E;margin-bottom:20px;
    flex-wrap:wrap;gap:16px">
    <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <h1 style="font-size:28px;font-weight:800;color:#FFFFFF;margin:0;
                       letter-spacing:-0.5px">{full_name}</h1>
            <span style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:6px;
                         padding:3px 10px;font-size:12px;color:#8E8E93;font-weight:600;
                         font-family:'JetBrains Mono',monospace">{ticker}</span>
        </div>
        <p style="color:#8E8E93;font-size:13px;margin:0">
            {sector} · {industry} · {country}
            {"· <a href='" + website + "' target='_blank' style='color:#0A84FF;text-decoration:none'>" + website.replace("https://","").replace("http://","").rstrip("/") + "</a>" if website else ""}
        </p>
    </div>
    <div style="text-align:right">
        <p style="font-size:36px;font-weight:800;color:#FFFFFF;margin:0;
                  letter-spacing:-1px">{fmt_price(price)}</p>
        <p style="color:{change_color};font-size:15px;font-weight:600;margin:4px 0 0">
            {change_arrow} {fmt_price(abs(change))} ({change_pct:+.2f}%)
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Top KPI strip ─────────────────────────────────────────────────────────────
strip_cols = st.columns(5)
strip_metrics = [
    ("Market Cap", fmt_large(market_cap)),
    ("Enterprise Value", fmt_large(enterprise_val)),
    ("52W High", fmt_price(price_data.get("high_52w"))),
    ("52W Low", fmt_price(price_data.get("low_52w"))),
    ("Employees", f"{employees:,}" if employees else "N/A"),
]
for col, (label, val) in zip(strip_cols, strip_metrics):
    col.metric(label, val)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─── Navigation tabs ───────────────────────────────────────────────────────────
tabs = st.tabs([
    "Price & Charts",
    "Financials",
    "KPIs & Health",
    "Insights",
    "News",
    "AI Copilot",
    "Peer Compare",
    "Survival Predictor",
    "Segments",
    "Private Co. Analysis",
    "IPO Tracker",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PRICE & CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # Quick price metrics
    pm_cols = st.columns(4)
    pm_cols[0].metric("Current Price", fmt_price(price),
                      f"{change_pct:+.2f}%",
                      delta_color="normal" if change >= 0 else "inverse")
    pm_cols[1].metric("Beta", fmt_multiple(price_data.get("beta")))
    pm_cols[2].metric("P/E Ratio", fmt_multiple(price_data.get("pe_ratio")))
    pm_cols[3].metric("Dividend Yield",
                      fmt_pct((price_data.get("dividend_yield") or 0) * 100, 2))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    period_choice = st.radio(
        "Time Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
        horizontal=True,
        index=3,
    )

    hist = get_price_history(ticker, period_choice)
    price_chart = build_price_chart(hist, ticker, period_choice)
    st.plotly_chart(price_chart, use_container_width=True, config=chart_config())

    # Performance stats for period
    if not hist.empty:
        period_start = hist["Close"].iloc[0]
        period_end = hist["Close"].iloc[-1]
        period_return = (period_end - period_start) / period_start * 100
        period_high = hist["High"].max()
        period_low = hist["Low"].min()
        avg_vol = hist["Volume"].mean()

        st.markdown("**Period Statistics**")
        ps_cols = st.columns(4)
        ps_cols[0].metric("Period Return", fmt_pct(period_return),
                          delta_color="normal" if period_return >= 0 else "inverse")
        ps_cols[1].metric("Period High", fmt_price(period_high))
        ps_cols[2].metric("Period Low", fmt_price(period_low))
        ps_cols[3].metric("Avg Daily Volume", fmt_large(avg_vol).replace("$", ""))

    # Revenue trend chart below price
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Revenue History**")
    rev_chart = build_revenue_trend_chart(income)
    if rev_chart.data:
        st.plotly_chart(rev_chart, use_container_width=True, config=chart_config())
    else:
        st.info("Revenue history data not available.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FINANCIAL STATEMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    stmt_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])

    def render_statement(df, label):
        if df is None or df.empty:
            st.warning(f"{label} data not available for {ticker}.")
            return
        formatted = format_statement_df(df)
        st.dataframe(
            formatted,
            use_container_width=True,
            height=min(600, max(200, len(formatted) * 35 + 50)),
        )
        # Download button
        csv_buf = io.StringIO()
        df.to_csv(csv_buf)
        st.download_button(
            label=f"Download {label} CSV",
            data=csv_buf.getvalue(),
            file_name=f"{ticker}_{label.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    with stmt_tabs[0]:
        st.markdown(f"**Annual Income Statement · Values in USD**")
        render_statement(income, "Income Statement")

    with stmt_tabs[1]:
        st.markdown(f"**Annual Balance Sheet · Values in USD**")
        render_statement(balance, "Balance Sheet")

    with stmt_tabs[2]:
        st.markdown(f"**Annual Cash Flow Statement · Values in USD**")
        render_statement(cashflow, "Cash Flow Statement")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: KPIs & FINANCIAL HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    # Health score header
    health_col, radar_col = st.columns([1, 1])

    with health_col:
        st.markdown("**Financial Health Score**")
        st.markdown(health_badge(health_label, health_score), unsafe_allow_html=True)

        # Dimension scores
        st.markdown("**Score Breakdown**")
        for dim, score in health_breakdown.items():
            pct = score / 20
            color = (COLORS["success"] if pct >= 0.75 else
                     COLORS["primary"] if pct >= 0.50 else
                     COLORS["warning"] if pct >= 0.25 else COLORS["danger"])
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="color:#FFFFFF;font-size:13px;font-weight:500">{dim}</span>
                    <span style="color:{color};font-size:13px;font-weight:600">{score}/20</span>
                </div>
                <div style="background:#2C2C2E;border-radius:4px;height:6px;width:100%">
                    <div style="background:{color};border-radius:4px;height:6px;
                                width:{pct*100:.0f}%;transition:width 0.5s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with radar_col:
        st.markdown("**Dimension Radar**")
        radar_fig = build_health_radar(health_breakdown)
        st.plotly_chart(radar_fig, use_container_width=True, config=chart_config())

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # KPI cards grid
    st.markdown("**Key Performance Indicators**")

    kpi_groups = {
        "Profitability": ["Revenue Growth %", "Gross Margin %", "Operating Margin %",
                          "Net Margin %", "EBITDA Margin %"],
        "Efficiency": ["ROA %", "ROE %", "FCF Margin %"],
        "Liquidity & Leverage": ["Current Ratio", "Quick Ratio", "Debt-to-Equity"],
    }

    for group_name, kpi_names in kpi_groups.items():
        st.markdown(f"**{group_name}**")
        cols = st.columns(len(kpi_names))
        for col, kpi_name in zip(cols, kpi_names):
            val, fmt_str, delta = kpis.get(kpi_name, (None, "N/A", None))
            delta_str = f"{delta:+.1f}pp" if delta is not None else None
            delta_color = ("positive" if delta and delta >= 0 else
                           "negative" if delta and delta < 0 else "normal")
            with col:
                st.metric(
                    kpi_name,
                    fmt_str,
                    delta=delta_str,
                    delta_color="normal" if delta_color == "positive" else
                                "inverse" if delta_color == "negative" else "off",
                )

    # Margin profile chart
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Margin Profile**")
    margin_chart = build_kpi_bar_chart(kpis)
    if margin_chart.data:
        st.plotly_chart(margin_chart, use_container_width=True, config=chart_config())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: EXECUTIVE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("**Executive Insights Panel**")
    st.markdown(
        '<p style="color:#8E8E93;font-size:13px">Automated intelligence generated from financial statements and KPI analysis.</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Generating insights..."):
        insights = generate_executive_insights(
            full_name, ticker, kpis, health_score, health_label, info,
            api_key=groq_key or None,
        )

    icons = {
        "Revenue Trend": "",
        "Profitability Trend": "",
        "Balance Sheet Strength": "",
        "Cash Flow Analysis": "",
    }
    for title, text in insights.items():
        icon = icons.get(title, "")
        st.markdown(f"""
        <div class="insight-card">
            <p style="color:#0A84FF;font-size:11px;text-transform:uppercase;
                      letter-spacing:0.8px;font-weight:700;margin:0 0 6px">{icon} {title}</p>
            <p style="color:#FFFFFF;font-size:14px;line-height:1.6;margin:0">{text}</p>
        </div>
        """, unsafe_allow_html=True)

    # CFO Brief section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**CFO Brief Generator**")
    st.markdown(
        '<p style="color:#8E8E93;font-size:13px">Generate a structured executive brief covering financial health, risks, growth drivers, and recommendations.</p>',
        unsafe_allow_html=True,
    )

    gen_col, _ = st.columns([1, 3])
    with gen_col:
        if st.button("Generate CFO Brief", use_container_width=True):
            with st.spinner("Compiling CFO Brief..."):
                news_items = get_company_news(ticker, full_name, limit=8)
                st.session_state.cfo_brief = generate_cfo_brief(
                    full_name, ticker, info, kpis, health_score, health_label,
                    news_items, api_key=groq_key or None,
                )

    if st.session_state.cfo_brief:
        st.markdown(
            f'<div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:24px">{st.session_state.cfo_brief}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download CFO Brief",
            data=st.session_state.cfo_brief,
            file_name=f"{ticker}_CFO_Brief_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )

    # Business Overview
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Business Overview**")
    with st.expander(f"About {full_name}", expanded=False):
        st.markdown(f'<p style="color:#FFFFFF;font-size:14px;line-height:1.7">{summary}</p>', unsafe_allow_html=True)
        meta_cols = st.columns(3)
        meta_cols[0].metric("Sector", sector)
        meta_cols[1].metric("Industry", industry)
        meta_cols[2].metric("Country", country)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: NEWS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    news_header_col, refresh_col = st.columns([3, 1])
    with news_header_col:
        st.markdown(f"**News · {full_name}**")
        st.markdown(
            '<p style="color:#8E8E93;font-size:13px">Latest news sorted by publication date. Click headlines to read full articles.</p>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        if st.button("Refresh News"):
            st.session_state.news_refresh += 1
            get_company_news.clear()

    with st.spinner("Fetching latest news..."):
        articles = get_company_news(ticker, full_name, limit=12)

    if not articles:
        st.info("No news articles found for this company. Try refreshing or check your connection.")
    else:
        for article in articles:
            title = article.get("title", "")
            publisher = article.get("publisher", "Unknown")
            pub_date = article.get("published_at", "")
            link = article.get("link", "#")

            st.markdown(f"""
            <div class="news-card">
                <a href="{link}" target="_blank" style="text-decoration:none">
                    <p style="color:#FFFFFF;font-size:14px;font-weight:600;
                              margin:0 0 8px;line-height:1.4">{title}</p>
                </a>
                <div style="display:flex;gap:12px;align-items:center">
                    <span style="background:#2C2C2E;border-radius:5px;
                                 padding:2px 8px;font-size:11px;color:#8E8E93;
                                 font-weight:500">{publisher}</span>
                    <span style="color:#48484A;font-size:12px">{pub_date}</span>
                    <a href="{link}" target="_blank" style="color:#0A84FF;
                       font-size:12px;text-decoration:none;margin-left:auto">Read →</a>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: AI COPILOT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    ai_left, ai_right = st.columns([2, 1])

    with ai_left:
        st.markdown(f"**AI Copilot · {full_name}**")
        st.markdown(
            f'<p style="color:#8E8E93;font-size:13px">Ask me anything about {full_name}\'s financials, risks, growth outlook, or strategy.</p>',
            unsafe_allow_html=True,
        )

        if not groq_key:
            st.markdown("""
            <div style="background:#FF9F0A11;border:1px solid #FF9F0A33;
                         border-radius:10px;padding:12px 16px;margin-bottom:16px">
                <p style="color:#FF9F0A;font-size:13px;margin:0;font-weight:500">
                    AI analysis is configured by the application when available.
                    Rule-based answers available without a key.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Display chat history
        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            suggested = [
                f"What are the biggest risks for {full_name}?",
                f"Summarize {full_name}'s latest financial performance.",
                f"What are {full_name}'s growth opportunities?",
                f"Analyze {full_name}'s financial health score.",
            ]
            sq_cols = st.columns(2)
            for i, q in enumerate(suggested):
                with sq_cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        news_items = get_company_news(ticker, full_name, limit=5)
                        headlines = [n["title"] for n in news_items]
                        response = chat_with_analyst(
                            q, full_name, ticker, info, kpis, health_score,
                            health_label, headlines,
                            api_key=groq_key or None,
                            chat_history=st.session_state.chat_history[:-1],
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.rerun()

        # Chat input
        if prompt := st.chat_input(f"Ask about {full_name}..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    news_items = get_company_news(ticker, full_name, limit=5)
                    headlines = [n["title"] for n in news_items]
                    response = chat_with_analyst(
                        prompt, full_name, ticker, info, kpis, health_score,
                        health_label, headlines,
                        api_key=groq_key or None,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        if st.session_state.chat_history:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    with ai_right:
        st.markdown("**Context Loaded**")
        st.markdown(f"""
        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:16px">
            <p style="color:#8E8E93;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;margin:0 0 12px">The AI has access to:</p>
            {''.join([
                f'<p style="color:#FFFFFF;font-size:13px;margin:0 0 8px">{item}</p>'
                for item in [
                    f"{full_name} financial statements",
                    "11 calculated KPIs",
                    f"Health score: {health_score}/100",
                    "Company metadata",
                    "Latest news headlines",
                ]
            ])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Snapshot**")
        snap_items = [
            ("Revenue Growth", kpis.get("Revenue Growth %", (None, "N/A"))[1]),
            ("Net Margin", kpis.get("Net Margin %", (None, "N/A"))[1]),
            ("Gross Margin", kpis.get("Gross Margin %", (None, "N/A"))[1]),
            ("ROE", kpis.get("ROE %", (None, "N/A"))[1]),
            ("FCF Margin", kpis.get("FCF Margin %", (None, "N/A"))[1]),
        ]
        for label, val in snap_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                padding:8px 0;border-bottom:1px solid #1C1C1E">
                <span style="color:#8E8E93;font-size:12px">{label}</span>
                <span style="color:#FFFFFF;font-size:12px;font-weight:600">{val}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: PEER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("**Peer Comparison**")

    # Get peer tickers
    peer_tickers = get_peers(ticker, info)

    # Allow custom peer input
    custom_peers = st.text_input(
        "Customize peers (comma-separated tickers)",
        value=", ".join(peer_tickers) if peer_tickers else "",
        placeholder="MSFT, GOOGL, AMZN",
    )
    if custom_peers:
        peer_tickers = [t.strip().upper() for t in custom_peers.split(",") if t.strip()]

    if not peer_tickers:
        st.info("No peer tickers identified. Enter competitor ticker symbols above.")
    else:
        with st.spinner(f"Loading peer data for {', '.join(peer_tickers)}..."):
            # Collect peer data
            peers_display = []
            comparison_data = {}

            # Current company
            comparison_data[ticker] = {
                "name": full_name,
                "ticker": ticker,
                "Market Cap": info.get("marketCap", 0) or 0,
                "Revenue": None,
                "Net Income": None,
                "Operating Margin": kpis.get("Operating Margin %", (None,))[0],
                "Net Margin": kpis.get("Net Margin %", (None,))[0],
                "ROE": kpis.get("ROE %", (None,))[0],
                "Gross Margin": kpis.get("Gross Margin %", (None,))[0],
            }

            # Revenue from income statement
            rev_keys = ["Total Revenue", "Revenue", "Net Revenue"]
            for k in rev_keys:
                if income is not None and not income.empty and k in income.index:
                    comparison_data[ticker]["Revenue"] = float(income.loc[k].iloc[0])
                    ni_keys = ["Net Income", "Net Income Common Stockholders"]
                    for nk in ni_keys:
                        if nk in income.index:
                            comparison_data[ticker]["Net Income"] = float(income.loc[nk].iloc[0])
                            break
                    break

            for pt in peer_tickers:
                try:
                    p_info = get_company_info(pt)
                    p_income = get_income_statement(pt)
                    p_kpis = calculate_kpis(p_income, get_balance_sheet(pt), get_cash_flow(pt), p_info)

                    p_rev, p_ni = None, None
                    for k in rev_keys:
                        if p_income is not None and not p_income.empty and k in p_income.index:
                            p_rev = float(p_income.loc[k].iloc[0])
                            for nk in ["Net Income", "Net Income Common Stockholders"]:
                                if nk in p_income.index:
                                    p_ni = float(p_income.loc[nk].iloc[0])
                                    break
                            break

                    comparison_data[pt] = {
                        "name": p_info.get("shortName") or pt,
                        "ticker": pt,
                        "Market Cap": p_info.get("marketCap", 0) or 0,
                        "Revenue": p_rev,
                        "Net Income": p_ni,
                        "Operating Margin": p_kpis.get("Operating Margin %", (None,))[0],
                        "Net Margin": p_kpis.get("Net Margin %", (None,))[0],
                        "ROE": p_kpis.get("ROE %", (None,))[0],
                        "Gross Margin": p_kpis.get("Gross Margin %", (None,))[0],
                    }
                    peers_display.append(comparison_data[pt])
                except Exception:
                    pass

        # Summary comparison table
        st.markdown("**Comparison Table**")
        table_rows = []
        for t, d in comparison_data.items():
            table_rows.append({
                "Company": d["name"],
                "Ticker": t,
                "Market Cap": fmt_large(d["Market Cap"]),
                "Revenue": fmt_large(d["Revenue"]) if d["Revenue"] else "N/A",
                "Net Income": fmt_large(d["Net Income"]) if d["Net Income"] else "N/A",
                "Gross Margin": fmt_pct(d["Gross Margin"]) if d["Gross Margin"] else "N/A",
                "Net Margin": fmt_pct(d["Net Margin"]) if d["Net Margin"] else "N/A",
                "ROE": fmt_pct(d["ROE"]) if d["ROE"] else "N/A",
            })
        comp_df = pd.DataFrame(table_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Chart comparisons
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Visual Comparison**")

        chart_metric = st.selectbox(
            "Select metric to chart",
            ["Market Cap", "Net Margin", "Gross Margin", "Operating Margin", "ROE"],
        )

        metric_key_map = {
            "Market Cap": "Market Cap",
            "Net Margin": "Net Margin",
            "Gross Margin": "Gross Margin",
            "Operating Margin": "Operating Margin",
            "ROE": "ROE",
        }

        all_tickers = [ticker] + [d["ticker"] for d in peers_display]
        all_vals = [comparison_data.get(t, {}).get(metric_key_map[chart_metric]) for t in all_tickers]
        all_names = [comparison_data.get(t, {}).get("name", t) for t in all_tickers]

        clean_names, clean_vals = [], []
        for name, val in zip(all_names, all_vals):
            if val is not None:
                clean_names.append(name)
                clean_vals.append(val)

        if clean_vals:
            bar_colors = [COLORS["primary"] if n == full_name else COLORS["neutral"] for n in clean_names]
            suffix = "%" if chart_metric != "Market Cap" else ""
            text_fn = [fmt_large(v) if chart_metric == "Market Cap" else f"{v:.1f}%" for v in clean_vals]

            fig = go.Figure(go.Bar(
                x=clean_names,
                y=clean_vals,
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=text_fn,
                textposition="outside",
                textfont=dict(color=COLORS["text"], size=12),
            ))
            layout = layout_defaults(f"{chart_metric} Comparison", height=350)
            layout["yaxis"]["ticksuffix"] = suffix
            layout["showlegend"] = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True, config=chart_config())
        else:
            st.info("Chart data not available for selected metric.")




# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: SURVIVAL PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("**Startup Financial Survival Predictor - Will This Company Survive the Next 24 Months?**")
    st.markdown("""
    <div style="background:#BF5AF211;border:1px solid #BF5AF233;border-radius:10px;
                padding:12px 16px;margin-bottom:20px">
        <p style="color:#BF5AF2;font-size:12px;font-weight:600;margin:0 0 4px">Methodology</p>
        <p style="color:#8E8E93;font-size:12px;margin:0;line-height:1.6">
            Built on distress pattern recognition from PRA Group debt portfolio analysis - identifying
            the financial fingerprint of companies 12-18 months before they fail. Combined with a
            Modified Altman Z-Score recalibrated for high-growth tech companies, cash runway modeling,
            revenue momentum decay analysis, and leverage risk scoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running survival model..."):
        sv = predict_survival(full_name, ticker, income, balance, cashflow, info)

    # ── Headline verdict ──────────────────────────────────────────────────────
    label_colors = {
        "Thriving":   COLORS["success"],
        "Vulnerable": COLORS["warning"],
        "Critical":   COLORS["danger"],
    }
    lc = label_colors.get(sv.overall_label, COLORS["neutral"])

    st.markdown(f"""
    <div style="background:{lc}11;border:2px solid {lc}44;border-radius:14px;
                padding:20px 24px;margin-bottom:24px">
        <p style="color:{lc};font-size:11px;text-transform:uppercase;
                  letter-spacing:1px;font-weight:700;margin:0 0 6px">24-Month Survival Verdict</p>
        <p style="color:#FFFFFF;font-size:20px;font-weight:700;margin:0 0 8px;line-height:1.4">{sv.headline}</p>
        <span style="background:{lc}33;color:{lc};font-size:13px;font-weight:700;
                     padding:4px 14px;border-radius:20px">{sv.overall_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Scenario probability distribution ─────────────────────────────────────
    st.markdown("**Scenario Probability Distribution**")
    p1, p2, p3 = st.columns(3)
    scenarios = [
        ("Scenario A", "Thriving", sv.prob_thriving,   COLORS["success"]),
        ("Scenario B", "Vulnerable", sv.prob_vulnerable, COLORS["warning"]),
        ("Scenario C", "Critical",  sv.prob_critical,   COLORS["danger"]),
    ]
    for col, (label, name, pct, color) in zip([p1, p2, p3], scenarios):
        with col:
            st.markdown(f"""
            <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;
                        padding:18px;text-align:center">
                <p style="color:#8E8E93;font-size:11px;margin:0 0 4px;text-transform:uppercase;
                          letter-spacing:0.5px">{label}</p>
                <p style="color:{color};font-size:14px;font-weight:600;margin:0 0 8px">{name}</p>
                <p style="color:{color};font-size:40px;font-weight:800;margin:0">{pct:.0f}%</p>
                <div style="background:#2C2C2E;border-radius:4px;height:6px;margin-top:10px">
                    <div style="background:{color};border-radius:4px;height:6px;
                                width:{min(pct,100):.0f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Sub-model scores ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    z_color = (COLORS["success"] if sv.z_score_label == "Safe Zone"
               else COLORS["warning"] if sv.z_score_label == "Grey Zone"
               else COLORS["danger"])
    rm = sv.runway_months
    rm_color = (COLORS["success"] if rm and rm > 24 else
                COLORS["warning"] if rm and rm > 12 else
                COLORS["danger"] if rm else COLORS["neutral"])

    with m1:
        st.markdown(f"""
        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:16px;text-align:center">
            <p style="color:#8E8E93;font-size:11px;margin:0 0 4px;text-transform:uppercase">Altman Z-Score</p>
            <p style="color:{z_color};font-size:28px;font-weight:800;margin:0">{sv.z_score:.2f}</p>
            <p style="color:{z_color};font-size:12px;font-weight:600;margin:4px 0 0">{sv.z_score_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        rm_str = f"{rm:.0f} mo" if rm else "N/A"
        st.markdown(f"""
        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:16px;text-align:center">
            <p style="color:#8E8E93;font-size:11px;margin:0 0 4px;text-transform:uppercase">Cash Runway</p>
            <p style="color:{rm_color};font-size:28px;font-weight:800;margin:0">{rm_str}</p>
            <p style="color:{rm_color};font-size:12px;font-weight:600;margin:4px 0 0">{sv.runway_label}</p>
        </div>
        """, unsafe_allow_html=True)

    mom_color = (COLORS["success"] if sv.momentum_score >= 65 else
                 COLORS["warning"] if sv.momentum_score >= 40 else COLORS["danger"])
    with m3:
        st.markdown(f"""
        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:16px;text-align:center">
            <p style="color:#8E8E93;font-size:11px;margin:0 0 4px;text-transform:uppercase">Rev Momentum</p>
            <p style="color:{mom_color};font-size:28px;font-weight:800;margin:0">{sv.momentum_score:.0f}/100</p>
            <p style="color:{mom_color};font-size:12px;font-weight:600;margin:4px 0 0">{sv.momentum_label}</p>
        </div>
        """, unsafe_allow_html=True)

    lev_color = (COLORS["success"] if sv.leverage_risk < 30 else
                 COLORS["warning"] if sv.leverage_risk < 60 else COLORS["danger"])
    mt_color = (COLORS["success"] if sv.margin_trajectory == "Improving" else
                COLORS["warning"] if "Stable" in sv.margin_trajectory else COLORS["danger"])
    with m4:
        st.markdown(f"""
        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:16px;text-align:center">
            <p style="color:#8E8E93;font-size:11px;margin:0 0 4px;text-transform:uppercase">Margin Trend</p>
            <p style="color:{mt_color};font-size:20px;font-weight:800;margin:0">{sv.margin_trajectory}</p>
            <p style="color:#8E8E93;font-size:11px;margin:4px 0 0">Leverage risk: {sv.leverage_risk:.0f}/100</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Signals ───────────────────────────────────────────────────────────────
    rf_col, gf_col = st.columns(2)
    with rf_col:
        st.markdown("**Distress Signals**")
        if sv.red_flags:
            for flag in sv.red_flags:
                st.markdown(f"""
                <div style="background:#FF3B3011;border:1px solid #FF3B3033;border-radius:8px;
                            padding:10px 14px;margin-bottom:6px">
                    <p style="color:#FF3B30;font-size:12px;margin:0;line-height:1.5">{flag}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#34C75911;border:1px solid #34C75933;border-radius:8px;padding:10px 14px"><p style="color:#34C759;font-size:13px;margin:0">✓ No critical distress signals detected</p></div>', unsafe_allow_html=True)

    with gf_col:
        st.markdown("**✅ Strength Signals**")
        if sv.green_flags:
            for flag in sv.green_flags:
                st.markdown(f"""
                <div style="background:#34C75911;border:1px solid #34C75933;border-radius:8px;
                            padding:10px 14px;margin-bottom:6px">
                    <p style="color:#34C759;font-size:12px;margin:0;line-height:1.5">{flag}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#FF9F0A11;border:1px solid #FF9F0A33;border-radius:8px;padding:10px 14px"><p style="color:#FF9F0A;font-size:13px;margin:0">No strong positive signals at this time</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Scenario narratives ───────────────────────────────────────────────────
    st.markdown("**Detailed Scenario Analysis**")
    for s_label, s_text, s_color in [
        ("Scenario A - Thriving", sv.scenario_a_text, COLORS["success"]),
        ("Scenario B - Vulnerable", sv.scenario_b_text, COLORS["warning"]),
        ("Scenario C - Critical", sv.scenario_c_text, COLORS["danger"]),
    ]:
        st.markdown(f"""
        <div style="background:#1C1C1E;border-left:3px solid {s_color};border-radius:0 10px 10px 0;
                    padding:14px 18px;margin-bottom:10px">
            <p style="color:{s_color};font-size:11px;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.8px;margin:0 0 6px">{s_label}</p>
            <p style="color:#FFFFFF;font-size:13px;line-height:1.7;margin:0">{s_text}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#2C2C2E22;border-radius:8px;padding:10px 14px;margin-top:8px">
        <p style="color:#48484A;font-size:11px;margin:0">
            Note: Quantitative model only. Not investment advice. Does not account for management quality,
            market conditions, regulatory changes, or strategic pivots.
            Methodology: Modified Altman Z-Score (recalibrated for tech) · Cash runway modeling ·
            Revenue momentum decay · Distress pattern recognition from debt portfolio analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9: PRIVATE COMPANY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("**Private Company Financial Analysis**")
    st.markdown(
        '<p style="color:#8E8E93;font-size:13px">Upload your financials to get KPI analysis, '
        'health scoring, AI insights, CFO Brief, and peer benchmarking against any public company.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:20px;margin-bottom:20px">
        <p style="color:#FFFFFF;font-size:14px;font-weight:600;margin:0 0 8px">Supported Formats</p>
        <p style="color:#8E8E93;font-size:13px;margin:0;line-height:1.8">
            <b style="color:#34C759">Excel (.xlsx)</b> - Name sheets: "Income Statement", "Balance Sheet", "Cash Flow"<br>
            <b style="color:#34C759">CSV (.csv)</b> - Single statement, first column as row labels, year columns as headers<br>
            <b style="color:#34C759">PDF (.pdf)</b> - Text-based annual reports with financial tables
        </p>
    </div>
    """, unsafe_allow_html=True)

    priv_left, priv_right = st.columns([1, 1])
    with priv_left:
        company_name_input = st.text_input("Company Name", placeholder="e.g. Acme Corp")
        uploaded_file = st.file_uploader(
            "Upload Financial Statements",
            type=["xlsx", "xls", "csv", "pdf"],
        )
    with priv_right:
        st.markdown("**Benchmark Against a Public Peer**")
        peer_input = st.text_input("Public company to compare against", placeholder="e.g. Microsoft, SAP")
        st.markdown("""
        <div style="background:#0A84FF0D;border:1px solid #0A84FF33;border-radius:10px;padding:14px;margin-top:8px">
            <p style="color:#0A84FF;font-size:12px;font-weight:600;margin:0 0 6px">Excel Format Tips</p>
            <p style="color:#8E8E93;font-size:12px;margin:0;line-height:1.6">
                First column: line item names (Revenue, Net Income, etc.)<br>
                Column headers: fiscal years (2022, 2023, 2024)<br>
                Values: actual units or millions - app auto-detects scale<br>
                Negatives: use minus sign or parentheses like (1,234)
            </p>
        </div>
        """, unsafe_allow_html=True)

    analyze_priv_btn = st.button("Analyze Private Company", key="priv_analyze")

    if uploaded_file and analyze_priv_btn:
        with st.spinner("Parsing financial statements..."):
            statements, detected_name = parse_uploaded_file(uploaded_file)
            priv_name = company_name_input.strip() or detected_name or "Private Company"
            available = get_available_statements(statements)

        if not available:
            st.error("Could not extract financial data. Check that rows are labeled (Revenue, Net Income, Total Assets) and columns are years (2022, 2023, 2024).")
        else:
            st.success(f"Parsed: {', '.join(available)} for **{priv_name}**")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            priv_kpis = calculate_kpis(statements["income"], statements["balance"], statements["cashflow"], {})
            priv_score, priv_label, priv_breakdown = calculate_health_score(priv_kpis, {})

            # Health Score
            score_col, radar_col = st.columns([1, 1])
            with score_col:
                st.markdown(f"**Financial Health Score**")
                st.markdown(health_badge(priv_label, priv_score), unsafe_allow_html=True)
                for dim, score in priv_breakdown.items():
                    pct = score / 20
                    color = (COLORS["success"] if pct >= 0.75 else COLORS["primary"] if pct >= 0.50
                             else COLORS["warning"] if pct >= 0.25 else COLORS["danger"])
                    st.markdown(f"""
                    <div style="margin-bottom:10px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span style="color:#FFFFFF;font-size:13px">{dim}</span>
                            <span style="color:{color};font-size:13px;font-weight:600">{score}/20</span>
                        </div>
                        <div style="background:#2C2C2E;border-radius:4px;height:6px">
                            <div style="background:{color};border-radius:4px;height:6px;width:{pct*100:.0f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            with radar_col:
                st.markdown("**Dimension Radar**")
                st.plotly_chart(build_health_radar(priv_breakdown), use_container_width=True, config=chart_config())

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # KPIs
            st.markdown("**Key Performance Indicators**")
            kpi_groups = {
                "Profitability": ["Revenue Growth %", "Gross Margin %", "Operating Margin %", "Net Margin %"],
                "Efficiency": ["ROA %", "ROE %", "FCF Margin %"],
                "Liquidity & Leverage": ["Current Ratio", "Quick Ratio", "Debt-to-Equity"],
            }
            for group_name, kpi_names in kpi_groups.items():
                st.markdown(f"**{group_name}**")
                cols = st.columns(len(kpi_names))
                for col, kpi_name in zip(cols, kpi_names):
                    _, fmt_str, _ = priv_kpis.get(kpi_name, (None, "N/A", None))
                    col.metric(kpi_name, fmt_str)

            # Statements
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Financial Statements**")
            stmt_map = {"Income Statement": "income", "Balance Sheet": "balance", "Cash Flow": "cashflow"}
            stmt_tab_labels = [s for s in stmt_map if s in available]
            if stmt_tab_labels:
                stmt_tabs_priv = st.tabs(stmt_tab_labels)
                for tab, label in zip(stmt_tabs_priv, stmt_tab_labels):
                    with tab:
                        df = statements[stmt_map[label]]
                        if not df.empty:
                            def fmt_priv(x):
                                try:
                                    v = float(x)
                                    if abs(v) >= 1e9: return f"${v/1e9:,.2f}B"
                                    if abs(v) >= 1e6: return f"${v/1e6:,.1f}M"
                                    return f"${v:,.0f}"
                                except: return "-"
                            try:
                                display_df = df.map(fmt_priv)
                            except AttributeError:
                                display_df = df.applymap(fmt_priv)
                            st.dataframe(display_df, use_container_width=True)
                            csv_buf = io.StringIO()
                            df.to_csv(csv_buf)
                            st.download_button(f"Download {label} CSV", csv_buf.getvalue(),
                                               f"{priv_name}_{label}.csv", "text/csv", key=f"dl_{label}")

            # Insights
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Executive Insights**")
            with st.spinner("Generating insights..."):
                priv_insights = generate_executive_insights(
                    priv_name, "PRIVATE", priv_kpis, priv_score, priv_label, {},
                    api_key=groq_key or None,
                )
            icons = {"Revenue Trend": "", "Profitability Trend": "", "Balance Sheet Strength": "", "Cash Flow Analysis": ""}
            for title, text in priv_insights.items():
                st.markdown(f"""
                <div class="insight-card">
                    <p style="color:#0A84FF;font-size:11px;text-transform:uppercase;
                              letter-spacing:0.8px;font-weight:700;margin:0 0 6px">{title}</p>
                    <p style="color:#FFFFFF;font-size:14px;line-height:1.6;margin:0">{text}</p>
                </div>
                """, unsafe_allow_html=True)

            # CFO Brief
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            if st.button("Generate CFO Brief", key="priv_cfo"):
                with st.spinner("Compiling CFO Brief..."):
                    priv_brief = generate_cfo_brief(
                        priv_name, "PRIVATE", {}, priv_kpis, priv_score, priv_label, [],
                        api_key=groq_key or None,
                    )
                st.markdown(f'<div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:24px">{priv_brief}</div>', unsafe_allow_html=True)
                st.download_button("Download CFO Brief", priv_brief,
                                   f"{priv_name}_CFO_Brief.md", "text/markdown", key="priv_dl_brief")

            # Peer comparison
            if peer_input.strip():
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown(f"**Peer Comparison: {priv_name} vs {peer_input}**")
                with st.spinner("Loading peer data..."):
                    peer_ticker, peer_full_name = resolve_ticker(peer_input.strip())
                if peer_ticker and peer_ticker != "PRIVATE":
                    peer_info = get_company_info(peer_ticker)
                    peer_kpis = calculate_kpis(
                        get_income_statement(peer_ticker),
                        get_balance_sheet(peer_ticker),
                        get_cash_flow(peer_ticker),
                        peer_info,
                    )
                    # Comparison table
                    compare_metrics = ["Revenue Growth %","Gross Margin %","Net Margin %",
                                       "Operating Margin %","ROE %","Current Ratio","Debt-to-Equity","FCF Margin %"]
                    rows = [{"Metric": m,
                             priv_name: priv_kpis.get(m,(None,"N/A"))[1],
                             peer_full_name: peer_kpis.get(m,(None,"N/A"))[1]}
                            for m in compare_metrics]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # Margin bar chart
                    margin_metrics = ["Gross Margin %","Operating Margin %","Net Margin %","FCF Margin %"]
                    labels, pvals, bvals = [], [], []
                    for m in margin_metrics:
                        pv = priv_kpis.get(m,(None,))[0]
                        bv = peer_kpis.get(m,(None,))[0]
                        if pv is not None and bv is not None:
                            labels.append(m.replace(" %",""))
                            pvals.append(pv)
                            bvals.append(bv)
                    if labels:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name=priv_name, x=labels, y=pvals,
                                             marker_color=COLORS["primary"],
                                             text=[f"{v:.1f}%" for v in pvals], textposition="outside"))
                        fig.add_trace(go.Bar(name=peer_full_name, x=labels, y=bvals,
                                             marker_color=COLORS["neutral"],
                                             text=[f"{v:.1f}%" for v in bvals], textposition="outside"))
                        layout = layout_defaults("Margin Comparison", height=350)
                        layout["barmode"] = "group"
                        layout["yaxis"]["ticksuffix"] = "%"
                        fig.update_layout(**layout)
                        st.plotly_chart(fig, use_container_width=True, config=chart_config())

                    # Health score side by side
                    pub_score, pub_label, _ = calculate_health_score(peer_kpis, peer_info)
                    h1, h2 = st.columns(2)
                    for col, name, sc, lb in [(h1, priv_name, priv_score, priv_label),
                                               (h2, peer_full_name, pub_score, pub_label)]:
                        hc = (COLORS["success"] if sc>=75 else COLORS["primary"] if sc>=55
                              else COLORS["warning"] if sc>=35 else COLORS["danger"])
                        with col:
                            st.markdown(f"""
                            <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;
                                        padding:20px;text-align:center">
                                <p style="color:#8E8E93;font-size:12px;margin:0 0 8px">{name}</p>
                                <p style="font-size:40px;font-weight:800;color:{hc};margin:0">{sc}</p>
                                <p style="color:{hc};font-size:14px;font-weight:600;margin:4px 0 0">{lb}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning(f"Could not find '{peer_input}'. Try a ticker like MSFT or AAPL.")

    elif not uploaded_file:
        # Sample template download
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**No file yet? Download a pre-formatted Excel template:**")
        sample = {
            "Income Statement": pd.DataFrame({
                "Line Item": ["Total Revenue","Gross Profit","Operating Income","Net Income","EBITDA"],
                "2022": [50e6, 20e6, 8e6, 5e6, 10e6],
                "2023": [60e6, 25e6, 10e6, 7e6, 13e6],
                "2024": [72e6, 31e6, 13e6, 9e6, 16e6],
            }),
            "Balance Sheet": pd.DataFrame({
                "Line Item": ["Total Current Assets","Total Assets","Total Current Liabilities","Total Debt","Total Stockholder Equity"],
                "2022": [15e6, 45e6, 8e6, 12e6, 25e6],
                "2023": [18e6, 52e6, 9e6, 10e6, 30e6],
                "2024": [22e6, 61e6, 10e6, 8e6, 37e6],
            }),
            "Cash Flow": pd.DataFrame({
                "Line Item": ["Operating Cash Flow","Capital Expenditure","Free Cash Flow"],
                "2022": [8e6, -2e6, 6e6],
                "2023": [11e6, -3e6, 8e6],
                "2024": [14e6, -3.5e6, 10.5e6],
            }),
        }
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, df in sample.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        output.seek(0)
        st.download_button(
            "Download Excel Template",
            data=output.getvalue(),
            file_name="FinIntel_Private_Company_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11: IPO FILING TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    render_ipo_tracker(compact=False)
