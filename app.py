import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import io
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

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinIntel AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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

/* Hide Streamlit default elements — keep header visible for sidebar toggle */
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


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0">
        <p style="font-size:22px;font-weight:800;color:#FFFFFF;margin:0;letter-spacing:-0.5px">
            FinIntel AI
        </p>
        <p style="font-size:12px;color:#8E8E93;margin:4px 0 0 0;font-weight:500">
            Financial Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode toggle ───────────────────────────────────────────────────────────
    st.markdown('<p style="color:#8E8E93;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">Analysis Mode</p>', unsafe_allow_html=True)
    app_mode = st.radio(
        "mode",
        ["Search Mode", "Upload Mode"],
        label_visibility="collapsed",
        horizontal=True,
    )
    st.markdown("---")

    # ── SEARCH MODE sidebar ───────────────────────────────────────────────────
    if app_mode == "Search Mode":
        st.markdown("**Search Company**")
        search_input = st.text_input(
            "Company name or ticker",
            placeholder="e.g. Microsoft, Apple, TCS...",
            label_visibility="collapsed",
        )
        search_col, _ = st.columns([1, 1])
        with search_col:
            search_btn = st.button("Analyze", use_container_width=True)

        st.markdown("---")
        st.markdown('<p style="color:#8E8E93;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px">Quick Access</p>', unsafe_allow_html=True)
        quick_companies = [
            ("MSFT", "Microsoft"), ("AAPL", "Apple"),
            ("NVDA", "NVIDIA"),    ("GOOGL", "Alphabet"),
            ("AMZN", "Amazon"),    ("TSLA", "Tesla"),
            ("SAP", "SAP SE"),     ("TCS.NS", "TCS"),
        ]
        cols = st.columns(2)
        for i, (t, name) in enumerate(quick_companies):
            with cols[i % 2]:
                if st.button(name, key=f"quick_{t}", use_container_width=True):
                    st.session_state.ticker = t
                    st.session_state.company_name = name
                    st.session_state.chat_history = []
                    st.session_state.cfo_brief = None
                    st.session_state.upload_statements = None
                    st.session_state.upload_company_name = None

        # ── Recently Analyzed ─────────────────────────────────────────────────
        if st.session_state.recent_companies:
            st.markdown("---")
            st.markdown('<p style="color:#8E8E93;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px">Recently Analyzed</p>', unsafe_allow_html=True)
            for r_ticker, r_name in st.session_state.recent_companies[:5]:
                if st.button(f"↩ {r_name}", key=f"recent_{r_ticker}",
                              use_container_width=True):
                    st.session_state.ticker = r_ticker
                    st.session_state.company_name = r_name
                    st.session_state.chat_history = []
                    st.session_state.cfo_brief = None
                    st.rerun()

    # ── UPLOAD MODE sidebar ───────────────────────────────────────────────────
    else:
        search_input = ""
        search_btn = False

        st.markdown("**Company Name**")
        upload_name_input = st.text_input(
            "Company name",
            placeholder="e.g. Acme Corp, My Division...",
            label_visibility="collapsed",
            key="upload_name",
        )

        st.markdown("**Upload Financials**")
        st.markdown('<p style="color:#8E8E93;font-size:11px;margin:0 0 6px">Excel, CSV, or PDF</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "financials",
            type=["xlsx", "xls", "csv", "pdf"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <div style="background:#0A84FF0D;border:1px solid #0A84FF33;border-radius:8px;
                    padding:10px 12px;margin:8px 0">
            <p style="color:#0A84FF;font-size:11px;font-weight:600;margin:0 0 4px">Excel Format</p>
            <p style="color:#8E8E93;font-size:11px;margin:0;line-height:1.6">
                Sheet names: Income Statement,<br>Balance Sheet, Cash Flow<br>
                Col headers: 2022, 2023, 2024<br>
                Row labels: Revenue, Net Income...
            </p>
        </div>
        """, unsafe_allow_html=True)

        upload_btn = st.button("Analyze Financials", use_container_width=True)

        # Benchmark peer input
        st.markdown("---")
        st.markdown("**Benchmark vs Public Peer**")
        st.markdown('<p style="color:#8E8E93;font-size:11px;margin:0 0 6px">Optional: compare against a public company</p>', unsafe_allow_html=True)
        upload_peer_input = st.text_input(
            "peer",
            placeholder="e.g. Microsoft, SAP...",
            label_visibility="collapsed",
            key="upload_peer",
        )

        # Template download
        st.markdown("---")
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
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sn, df in sample.items():
                df.to_excel(writer, sheet_name=sn, index=False)
        buf.seek(0)
        st.download_button(
            "Download Excel Template",
            data=buf.getvalue(),
            file_name="FinIntel_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Handle upload button
        if upload_btn and uploaded_file:
            with st.spinner("Parsing..."):
                stmts, detected = parse_uploaded_file(uploaded_file)
                uname = upload_name_input.strip() or detected or "My Company"
                avail = get_available_statements(stmts)
            if avail:
                st.session_state.upload_statements = stmts
                st.session_state.upload_company_name = uname
                st.session_state.upload_peer = upload_peer_input.strip()
                st.session_state.ticker = None  # clear search mode
                st.success(f"Parsed: {', '.join(avail)}")
            else:
                st.error("Could not extract data. Check row labels and year columns.")
        elif upload_btn and not uploaded_file:
            st.warning("Please upload a file first.")

    # ── AI Settings (both modes) ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p style="color:#8E8E93;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px">AI Settings</p>', unsafe_allow_html=True)
    _server_key = _load_groq_key()
    if _server_key:
        groq_key = _server_key
        st.markdown("""
        <div style="background:#34C75911;border:1px solid #34C75933;border-radius:8px;padding:10px 12px">
            <p style="color:#34C759;font-size:12px;font-weight:600;margin:0">AI Analysis Active</p>
            <p style="color:#8E8E93;font-size:11px;margin:4px 0 0">Powered by Groq · Llama 3.3 70B</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
            label_visibility="collapsed",
        )
        if groq_key:
            st.success("AI mode enabled")
        else:
            st.markdown("""
            <div style="background:#FF9F0A11;border:1px solid #FF9F0A33;border-radius:8px;padding:10px 12px">
                <p style="color:#FF9F0A;font-size:11px;font-weight:500;margin:0">
                    AI features need a Groq key.<br>
                    Get one free at <a href="https://console.groq.com" target="_blank"
                    style="color:#FF9F0A">console.groq.com</a>
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style="color:#48484A;font-size:10px;text-align:center">
        Data via Yahoo Finance · Not financial advice<br>
        Built by Hetal Shah · github.com/Hshah168
    </p>
    """, unsafe_allow_html=True)

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
    else:
        st.error(f"Could not identify a publicly traded company for **{search_input}**. Try a ticker directly.")


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD MODE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Upload Mode":
    if not st.session_state.upload_statements:

        st.markdown("""
        <div style="text-align:center;padding:60px 20px 30px">
            <h1 style="font-size:32px;font-weight:800;color:#FFFFFF;margin:12px 0 8px">
                Upload Mode
            </h1>

            <p style="font-size:16px;color:#8E8E93;margin:0 0 12px;max-width:500px;display:inline-block">
                Analyze any company's internal financials. Public or private,<br>
                listed or unlisted — if you have the numbers, we can analyze them.
            </p>
        </div>
        """, unsafe_allow_html=True)

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

        for col, item in zip(use_cols, use_cases):
            title, desc = item

            with col:
                st.markdown(f"""
                <div style="
                    background:#1C1C1E;
                    border:1px solid #2C2C2E;
                    border-radius:14px;
                    padding:20px;
                    text-align:center;
                    min-height:140px;
                ">
                    <p style="font-size:14px;font-weight:700;color:#FFFFFF;">
                        {title}
                    </p>

                    <p style="font-size:12px;color:#8E8E93;line-height:1.5;">
                        {desc}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <p style="text-align:center;color:#48484A;font-size:13px;margin-top:24px">
        Upload your file and click <b>Analyze Financials</b> in the sidebar to begin.
        </p>
        """, unsafe_allow_html=True)

        st.stop()

    # ── Upload mode has data — run full analysis ──────────────────────────────
    stmts = st.session_state.upload_statements
    uname = st.session_state.upload_company_name
    upeer = st.session_state.upload_peer

    u_kpis = calculate_kpis(stmts["income"], stmts["balance"], stmts["cashflow"], {})
    u_score, u_label, u_breakdown = calculate_health_score(u_kpis, {})

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
        padding:20px 0 16px;border-bottom:1px solid #1C1C1E;margin-bottom:20px;flex-wrap:wrap;gap:16px">
        <div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <h1 style="font-size:28px;font-weight:800;color:#FFFFFF;margin:0">{uname}</h1>
                <span style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:6px;
                             padding:3px 10px;font-size:12px;color:#8E8E93;font-weight:600">UPLOADED</span>
            </div>
            <p style="color:#8E8E93;font-size:13px;margin:0">Internal financial data · Confidential analysis</p>
        </div>
        <div style="text-align:right">
            <p style="font-size:14px;color:#8E8E93;margin:0">Financial Health</p>
            <p style="font-size:32px;font-weight:800;color:{'#34C759' if u_score>=75 else '#0A84FF' if u_score>=55 else '#FF9F0A' if u_score>=35 else '#FF3B30'};margin:0">{u_score}/100</p>
            <p style="color:#8E8E93;font-size:13px;margin:0">{u_label}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                        except: return "—"
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
if not st.session_state.ticker:
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 32px">
        <h1 style="font-size:48px;font-weight:800;color:#FFFFFF;margin:0 0 16px;
                   letter-spacing:-1px">FinIntel AI</h1>
        <p style="font-size:18px;color:#8E8E93;margin:0 0 0;max-width:520px;
                   display:inline-block;line-height:1.6">
            Enterprise-grade financial intelligence for analysts,<br>
            FP&A teams, and business leaders.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Capability strip ──────────────────────────────────────────────────────
    for col, (icon, title, sub) in zip(
        st.columns(4),
        [("", "Smart Search",     "Type any company name, no ticker needed"),
         ("", "Variance Analysis", "Automated YoY variance with plain-English narrative"),
         ("", "Upload Mode",       "Analyze internal financials not in public filings"),
         ("", "Survival Predictor","24-month distress model from PRA Group methodology")],
    ):
        with col:
            st.markdown(f"""
            <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:10px;
                        padding:14px 16px;">
                <span style="font-size:20px">{icon}</span>
                <p style="color:#FFFFFF;font-size:13px;font-weight:600;margin:8px 0 2px">{title}</p>
                <p style="color:#8E8E93;font-size:11px;margin:0">{sub}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    # ── What to do next ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0A84FF0D;border:1px solid #0A84FF33;border-radius:12px;
                padding:20px 24px;margin-bottom:24px">
        <p style="color:#0A84FF;font-size:13px;font-weight:600;margin:0 0 10px">
            How to get started
        </p>
        <div style="display:flex;gap:32px;flex-wrap:wrap">
            <div>
                <p style="color:#FFFFFF;font-size:13px;font-weight:600;margin:0 0 2px">
                    Search Mode
                </p>
                <p style="color:#8E8E93;font-size:12px;margin:0">
                    Type any company name in the sidebar.<br>
                    Works for public companies globally.
                </p>
            </div>
            <div>
                <p style="color:#FFFFFF;font-size:13px;font-weight:600;margin:0 0 2px">
                    Upload Mode
                </p>
                <p style="color:#8E8E93;font-size:12px;margin:0">
                    Toggle Upload Mode in the sidebar.<br>
                    Drop in your Excel, CSV, or PDF.
                </p>
            </div>
            <div>
                <p style="color:#FFFFFF;font-size:13px;font-weight:600;margin:0 0 2px">
                    Quick Access
                </p>
                <p style="color:#8E8E93;font-size:12px;margin:0">
                    Use the quick access buttons in the sidebar<br>
                    to instantly load major companies.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── What you get strip ────────────────────────────────────────────────────
    st.markdown('<p style="color:#8E8E93;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin:0 0 12px">What you get for every company</p>', unsafe_allow_html=True)
    features = [
        ("11 KPIs",            "Calculated automatically from live financial statements"),
        ("Health Score",       "0-100 score across profitability, growth, liquidity, leverage, cash flow"),
        ("Variance Analysis",  "YoY change with plain-English narrative for every major line item"),
        ("CFO Brief",          "One-click structured brief — download as PDF or PowerPoint"),
        ("Segment Breakdown",  "Revenue by business division for major companies"),
        ("Survival Predictor", "24-month probabilistic distress model"),
        ("Upload Mode",        "Same analysis on your own private financials"),
        ("Peer Benchmark",     "Side-by-side comparison against any public competitor"),
    ]
    r1, r2 = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with (r1 if i % 2 == 0 else r2):
            st.markdown(f"""
            <div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #1C1C1E">
                <span style="color:#0A84FF;font-size:14px;margin-top:1px">✓</span>
                <div>
                    <span style="color:#FFFFFF;font-size:13px;font-weight:600">{title}</span>
                    <span style="color:#8E8E93;font-size:12px;margin-left:6px">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#48484A;font-size:11px;margin-top:20px;text-align:center">
        Data via Yahoo Finance · Not financial advice · Built by Hetal Shah · github.com/Hshah168
    </p>
    """, unsafe_allow_html=True)
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
                    Tip: Add your Groq API key in the sidebar for full AI responses.
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
    st.markdown("**Startup Financial Survival Predictor — Will This Company Survive the Next 24 Months?**")
    st.markdown("""
    <div style="background:#BF5AF211;border:1px solid #BF5AF233;border-radius:10px;
                padding:12px 16px;margin-bottom:20px">
        <p style="color:#BF5AF2;font-size:12px;font-weight:600;margin:0 0 4px">Methodology</p>
        <p style="color:#8E8E93;font-size:12px;margin:0;line-height:1.6">
            Built on distress pattern recognition from PRA Group debt portfolio analysis — identifying
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
        ("Scenario A — Thriving", sv.scenario_a_text, COLORS["success"]),
        ("Scenario B — Vulnerable", sv.scenario_b_text, COLORS["warning"]),
        ("Scenario C — Critical", sv.scenario_c_text, COLORS["danger"]),
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
            <b style="color:#34C759">Excel (.xlsx)</b> — Name sheets: "Income Statement", "Balance Sheet", "Cash Flow"<br>
            <b style="color:#34C759">CSV (.csv)</b> — Single statement, first column as row labels, year columns as headers<br>
            <b style="color:#34C759">PDF (.pdf)</b> — Text-based annual reports with financial tables
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
                Values: actual units or millions — app auto-detects scale<br>
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
                                except: return "—"
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
