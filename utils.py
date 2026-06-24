"""
utils.py
--------
Shared formatting helpers, Plotly chart builders,
and Streamlit UI components.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─── Color palette ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#0A84FF",
    "success": "#34C759",
    "warning": "#FF9F0A",
    "danger": "#FF3B30",
    "neutral": "#636366",
    "bg_card": "#1C1C1E",
    "bg_page": "#000000",
    "text": "#FFFFFF",
    "text_muted": "#8E8E93",
    "border": "#2C2C2E",
    "chart_up": "#34C759",
    "chart_down": "#FF3B30",
    "chart_blue": "#0A84FF",
    "chart_purple": "#BF5AF2",
    "chart_orange": "#FF9F0A",
}


# ─── Number formatting ────────────────────────────────────────────────────────
def fmt_large(n: float | None) -> str:
    """Format large numbers as $1.23T / $456.7B / $78.9M."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    n = float(n)
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def fmt_price(n: float | None, decimals: int = 2) -> str:
    if n is None:
        return "N/A"
    return f"${float(n):,.{decimals}f}"


def fmt_pct(n: float | None, decimals: int = 2) -> str:
    if n is None:
        return "N/A"
    return f"{float(n):.{decimals}f}%"


def fmt_multiple(n: float | None) -> str:
    if n is None:
        return "N/A"
    return f"{float(n):.2f}x"


# ─── Plotly chart builders ─────────────────────────────────────────────────────
def chart_config() -> dict:
    """Standard Plotly config for all charts."""
    return {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "displaylogo": False,
    }


def layout_defaults(title: str = "", height: int = 380) -> dict:
    """Standard dark layout dict."""
    return dict(
        title=dict(text=title, font=dict(size=14, color=COLORS["text_muted"])),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
        xaxis=dict(
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickcolor=COLORS["text_muted"],
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickcolor=COLORS["text_muted"],
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )


def build_price_chart(df: pd.DataFrame, ticker: str, period_label: str) -> go.Figure:
    """Candlestick + volume chart for price history."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No price data available", showarrow=False,
                           font=dict(color=COLORS["text_muted"], size=14))
        fig.update_layout(**layout_defaults())
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color=COLORS["chart_up"],
            decreasing_line_color=COLORS["chart_down"],
            increasing_fillcolor=COLORS["chart_up"],
            decreasing_fillcolor=COLORS["chart_down"],
        ),
        row=1, col=1,
    )

    # 20-day MA overlay
    if len(df) >= 20:
        ma20 = df["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ma20, name="20d MA",
                line=dict(color=COLORS["chart_orange"], width=1.5, dash="dot"),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    # Volume bars
    colors = [
        COLORS["chart_up"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
        else COLORS["chart_down"]
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.6,
        ),
        row=2, col=1,
    )

    layout = layout_defaults(f"{ticker} · {period_label}", height=460)
    layout["xaxis2"] = dict(gridcolor=COLORS["border"], linecolor=COLORS["border"])
    layout["yaxis2"] = dict(
        title="Volume", gridcolor=COLORS["border"], linecolor=COLORS["border"]
    )
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    return fig


def build_kpi_bar_chart(kpi_data: dict, peers_data: dict | None = None) -> go.Figure:
    """Horizontal bar chart of key margins."""
    margin_kpis = ["Gross Margin %", "Operating Margin %", "Net Margin %", "FCF Margin %"]
    labels = []
    values = []
    for k in margin_kpis:
        val = kpi_data.get(k, (None,))[0]
        if val is not None:
            labels.append(k.replace(" %", ""))
            values.append(val)

    if not values:
        return go.Figure()

    bar_colors = [COLORS["chart_up"] if v >= 0 else COLORS["chart_down"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=12),
    ))
    layout = layout_defaults("Margin Profile", height=280)
    layout["xaxis"]["ticksuffix"] = "%"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def build_health_radar(breakdown: dict) -> go.Figure:
    """Radar chart of health score components."""
    categories = list(breakdown.keys())
    values = list(breakdown.values())
    max_v = 20  # each dimension max 20

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=f"rgba(10, 132, 255, 0.2)",
        line=dict(color=COLORS["primary"], width=2),
        name="Score",
        hovertemplate="%{theta}: %{r}/20<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_v],
                tickfont=dict(size=10, color=COLORS["text_muted"]),
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["text"]),
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False,
    )
    return fig


def build_peer_comparison_chart(
    current_ticker: str,
    current_data: dict,
    peer_data: list[dict],
    metric: str,
) -> go.Figure:
    """Bar chart comparing a metric across peers."""
    tickers = [current_ticker] + [p["ticker"] for p in peer_data]
    values = [current_data.get(metric)] + [p.get(metric) for p in peer_data]

    clean_tickers, clean_values = [], []
    for t, v in zip(tickers, values):
        if v is not None:
            clean_tickers.append(t)
            clean_values.append(v)

    if not clean_values:
        return go.Figure()

    bar_colors = [
        COLORS["primary"] if t == current_ticker else COLORS["neutral"]
        for t in clean_tickers
    ]

    fig = go.Figure(go.Bar(
        x=clean_tickers,
        y=clean_values,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[fmt_large(v) if metric == "Market Cap" else f"{v:.1f}%" for v in clean_values],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    layout = layout_defaults(metric, height=300)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def build_revenue_trend_chart(income: pd.DataFrame) -> go.Figure:
    """Line chart of revenue over available fiscal years."""
    if income is None or income.empty:
        return go.Figure()

    rev_keys = ["Total Revenue", "Revenue", "Net Revenue", "Revenues"]
    rev_row = None
    for k in rev_keys:
        if k in income.index:
            rev_row = income.loc[k]
            break
    if rev_row is None:
        return go.Figure()

    years = [
        c.strftime("%Y") if hasattr(c, "strftime") else str(c)
        for c in rev_row.index
    ][::-1]
    values = [float(v) / 1e9 if pd.notna(v) else None for v in rev_row.values][::-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=values,
        mode="lines+markers+text",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8, color=COLORS["primary"]),
        text=[f"${v:.1f}B" if v else "" for v in values],
        textposition="top center",
        textfont=dict(color=COLORS["text"], size=11),
        fill="tozeroy",
        fillcolor=f"rgba(10, 132, 255, 0.1)",
        name="Revenue",
        hovertemplate="FY %{x}: $%{y:.2f}B<extra></extra>",
    ))
    layout = layout_defaults("Revenue Trend (USD Billions)", height=300)
    layout["yaxis"]["tickprefix"] = "$"
    layout["yaxis"]["ticksuffix"] = "B"
    fig.update_layout(**layout)
    return fig


# ─── Streamlit UI components ────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str | None = None,
             delta_color: str = "normal", icon: str = "") -> str:
    """Return an HTML KPI card string."""
    delta_html = ""
    if delta:
        color = (COLORS["success"] if delta_color == "positive"
                 else COLORS["danger"] if delta_color == "negative"
                 else COLORS["text_muted"])
        arrow = "▲" if delta_color == "positive" else "▼" if delta_color == "negative" else "●"
        delta_html = f'<p style="color:{color};font-size:12px;margin:4px 0 0 0">{arrow} {delta}</p>'

    return f"""
<div style="
    background:{COLORS['bg_card']};
    border:1px solid {COLORS['border']};
    border-radius:12px;
    padding:16px 18px;
    margin-bottom:12px;
    min-height:80px;
">
    <p style="color:{COLORS['text_muted']};font-size:11px;text-transform:uppercase;
              letter-spacing:0.8px;margin:0 0 6px 0;font-weight:600">{icon} {label}</p>
    <p style="color:{COLORS['text']};font-size:22px;font-weight:700;margin:0;
              line-height:1.2">{value}</p>
    {delta_html}
</div>
"""


def health_badge(label: str, score: int) -> str:
    color_map = {
        "Excellent": COLORS["success"],
        "Strong": COLORS["primary"],
        "Average": COLORS["warning"],
        "Weak": COLORS["danger"],
    }
    color = color_map.get(label, COLORS["neutral"])
    return f"""
<div style="display:inline-flex;align-items:center;gap:10px;
    background:{color}22;border:1px solid {color}44;
    border-radius:20px;padding:8px 20px;margin:8px 0">
    <span style="font-size:24px;font-weight:800;color:{color}">{score}</span>
    <div>
        <p style="color:{color};font-size:11px;text-transform:uppercase;
                  letter-spacing:1px;font-weight:700;margin:0">Financial Health</p>
        <p style="color:{COLORS['text']};font-size:16px;font-weight:600;margin:0">{label}</p>
    </div>
</div>
"""
