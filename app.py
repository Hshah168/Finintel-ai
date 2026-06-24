import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import io
from datetime import datetime

from company_search import resolve_ticker, get_company_info

# ─── Load Groq API key (secrets → env → sidebar input, in that order) ─────────
def _load_groq_key() -> str | None:
    """
    Priority order:
    1. Streamlit secrets  — .streamlit/secrets.toml (local) or Secrets panel (Cloud)
    2. Environment variable GROQ_API_KEY
    Returns None if no valid key found anywhere.
    """
    import os

    # 1. Streamlit secrets
    try:
        key = st.secrets["GROQ_API_KEY"]        
        if key and key.startswith("gsk_"):
            return key.strip()
    except Exception:
        pass

    # 2. Environment variable
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key.startswith("gsk_"):
        return key

    return None
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
    kpi_card, health_badge, chart_config, layout_defaults, COLORS,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinIntel AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",  # always open on load
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


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0">
        <p style="font-size:22px;font-weight:800;color:#FFFFFF;margin:0;letter-spacing:-0.5px">
             FinIntel AI
        </p>
        <p style="font-size:12px;color:#8E8E93;margin:4px 0 0 0;font-weight:500">
            Financial Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Search Company**")
    search_input = st.text_input(
        "Company name or ticker",
        placeholder="e.g. Microsoft, Apple, TCS...",
        label_visibility="collapsed",
    )

    search_col, _ = st.columns([1, 1])
    with search_col:
        search_btn = st.button(" Analyze", use_container_width=True)

    # Quick access companies
    st.markdown("---")
    st.markdown('<p style="color:#8E8E93;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px">Quick Access</p>', unsafe_allow_html=True)
    quick_companies = [
        ("MSFT", "Microsoft"),
        ("AAPL", "Apple"),
        ("NVDA", "NVIDIA"),
        ("GOOGL", "Alphabet"),
        ("AMZN", "Amazon"),
        ("TSLA", "Tesla"),
        ("SAP", "SAP SE"),
        ("TCS.NS", "TCS"),
    ]
    cols = st.columns(2)
    for i, (ticker, name) in enumerate(quick_companies):
        with cols[i % 2]:
            if st.button(name, key=f"quick_{ticker}", use_container_width=True):
                st.session_state.ticker = ticker
                st.session_state.company_name = name
                st.session_state.chat_history = []
                st.session_state.cfo_brief = None

    # AI Settings
    st.markdown("---")
    st.markdown('<p style="color:#8E8E93;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px">AI Settings</p>', unsafe_allow_html=True)

    # Try to load key from secrets/env first
    _server_key = _load_groq_key()

    if _server_key:
        # Key is embedded — AI is on for everyone, no input needed
        groq_key = _server_key
        st.markdown("""
        <div style="background:#34C75911;border:1px solid #34C75933;border-radius:8px;padding:10px 12px">
            <p style="color:#34C759;font-size:12px;font-weight:600;margin:0">🤖 AI Analysis Active</p>
            <p style="color:#8E8E93;font-size:11px;margin:4px 0 0">Powered by Groq · Llama 3.3 70B</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # No server key — show optional override input
        st.markdown('<p style="color:#8E8E93;font-size:11px;margin:0 0 6px">Optional: add your own Groq key for AI features</p>', unsafe_allow_html=True)
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
            label_visibility="collapsed",
        )
        if groq_key:
            st.success("✓ AI mode enabled", icon="🤖")
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


# ─── Search handler ────────────────────────────────────────────────────────────
if search_btn and search_input.strip():
    with st.spinner(f"Identifying {search_input}..."):
        ticker, full_name = resolve_ticker(search_input.strip())
    if ticker == "PRIVATE":
        st.error(f"**{search_input}** is a privately held company and is not listed on any public stock exchange. FinIntel AI analyzes publicly traded companies only.")
    elif ticker:
        st.session_state.ticker = ticker
        st.session_state.company_name = full_name
        st.session_state.chat_history = []
        st.session_state.cfo_brief = None
    else:
        st.error(f"Could not identify a publicly traded company for **{search_input}**. Try a ticker symbol directly (e.g. NSANY for Nissan, HMC for Honda).")


# ─── Landing page (no company selected) ───────────────────────────────────────
if not st.session_state.ticker:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px 40px">
        <p style="font-size:48px;margin:0">📊</p>
        <h1 style="font-size:40px;font-weight:800;color:#FFFFFF;margin:12px 0 8px;
                   letter-spacing:-1px">FinIntel AI</h1>
        <p style="font-size:18px;color:#8E8E93;margin:0 0 32px;max-width:520px;
                   display:inline-block">
            Enterprise-grade financial intelligence for analysts,<br>
            FP&A teams, and business leaders.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3)
    features = [
        ("🔍", "Smart Search", "Type any company name — Microsoft, TCS, SAP, Reliance. Auto-resolves to ticker."),
        ("📈", "Deep Analytics", "Income statements, balance sheets, cash flows, KPIs, and financial health scoring."),
        ("🤖", "AI Insights", "AI Analyst Copilot, CFO Brief generator, and executive insight narratives."),
        ("🌍", "Global Coverage", "US, Indian, European, and Asian markets. Supports NSE, NYSE, NASDAQ, and more."),
        ("⚡", "Peer Comparison", "Benchmark against competitors across revenue, margins, ROE, and market cap."),
        ("📰", "News Center", "Real-time financial news aggregated and ranked for any company."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:14px;
                        padding:20px;margin-bottom:16px;min-height:130px">
                <p style="font-size:28px;margin:0 0 8px">{icon}</p>
                <p style="font-size:14px;font-weight:700;color:#FFFFFF;margin:0 0 6px">{title}</p>
                <p style="font-size:12px;color:#8E8E93;margin:0;line-height:1.5">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:20px 0 40px">
        <p style="color:#48484A;font-size:13px">
            Search a company in the sidebar to begin · Powered by Yahoo Finance & Groq AI
        </p>
    </div>
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
    " Price & Charts",
    " Financials",
    " KPIs & Health",
    " Insights",
    " News",
    " AI Copilot",
    " Peer Compare",
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
            label=f"⬇ Download {label} CSV",
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
        "Revenue Trend": "📈",
        "Profitability Trend": "💰",
        "Balance Sheet Strength": "🏦",
        "Cash Flow Analysis": "💸",
    }
    for title, text in insights.items():
        icon = icons.get(title, "📊")
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
        if st.button(" Generate CFO Brief", use_container_width=True):
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
            "⬇ Download CFO Brief",
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
        st.markdown(f"**Financial News · {full_name}**")
        st.markdown(
            '<p style="color:#8E8E93;font-size:13px">Latest news sorted by publication date. Click headlines to read full articles.</p>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        if st.button(" Refresh News"):
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
                    <span style="color:#48484A;font-size:12px"> {pub_date}</span>
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
        st.markdown(f"**AI Analyst Copilot · {full_name}**")
        st.markdown(
            f'<p style="color:#8E8E93;font-size:13px">Ask me anything about {full_name}\'s financials, risks, growth outlook, or strategy.</p>',
            unsafe_allow_html=True,
        )

        if not groq_key:
            st.markdown("""
            <div style="background:#FF9F0A11;border:1px solid #FF9F0A33;
                         border-radius:10px;padding:12px 16px;margin-bottom:16px">
                <p style="color:#FF9F0A;font-size:13px;margin:0;font-weight:500">
                     Add your Groq API key in the sidebar for full AI responses.
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
            if st.button("🗑 Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    with ai_right:
        st.markdown("**Context Loaded**")
        st.markdown(f"""
        <div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;padding:16px">
            <p style="color:#8E8E93;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;margin:0 0 12px">The AI has access to:</p>
            {''.join([
                f'<p style="color:#FFFFFF;font-size:13px;margin:0 0 8px">✓ {item}</p>'
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
