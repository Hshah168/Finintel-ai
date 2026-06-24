import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


@st.cache_data(ttl=3600)
def get_income_statement(ticker: str) -> pd.DataFrame:
    """Annual income statement, most recent year first."""
    try:
        t = yf.Ticker(ticker)
        df = t.financials  # columns = fiscal year end dates
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_index(axis=1, ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_balance_sheet(ticker: str) -> pd.DataFrame:
    """Annual balance sheet, most recent year first."""
    try:
        t = yf.Ticker(ticker)
        df = t.balance_sheet
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_index(axis=1, ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_cash_flow(ticker: str) -> pd.DataFrame:
    """Annual cash flow statement, most recent year first."""
    try:
        t = yf.Ticker(ticker)
        df = t.cashflow
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_index(axis=1, ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Retrieve OHLCV price history.

    period options: '1mo', '3mo', '6mo', '1y', '5y'
    """
    period_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "5 Years": "5y",
    }
    yf_period = period_map.get(period, period)

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=yf_period, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_current_price_data(ticker: str) -> dict:
    """
    Return a dict of key real-time / near-real-time price metrics.
    """
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "avg_volume": info.get("averageVolume"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {}


def format_statement_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format financial statement for display — currency in billions."""
    if df is None or df.empty:
        return df
    display = df.copy()
    # Format column headers as year strings
    display.columns = [
        c.strftime("%Y") if hasattr(c, "strftime") else str(c)
        for c in display.columns
    ]
    # Convert values to billions with 2 decimal places
    def fmt(x):
        if pd.isna(x):
            return "—"
        b = x / 1e9
        return f"${b:,.2f}B"

    try:
        return display.map(fmt)       # pandas >= 2.1
    except AttributeError:
        return display.applymap(fmt)  # pandas < 2.1 fallback


def get_peers(ticker: str, info: dict) -> list[str]:
    """
    Return a list of peer tickers for comparison.
    Falls back to sector-based peers if ticker not in map.
    """
    peer_map = {
        # ── Space / Aerospace ─────────────────────────────────────────────
        "SPCX": ["BA", "LMT", "RKLB"],       # SpaceX
        "RKLB": ["SPCX", "ASTR", "BA"],       # Rocket Lab
        # ── 2025-2026 IPOs ────────────────────────────────────────────────
        "CRWV": ["NVDA", "AMD", "CBRS"],      # CoreWeave — AI cloud
        "CBRS": ["NVDA", "CRWV", "AMD"],      # Cerebras — AI chips
        "CRCL": ["COIN", "PYPL", "V"],        # Circle Internet — crypto/stablecoin
        "FIG":  ["ADBE", "CRM", "NOW"],       # Figma — design platform
        "QNT":  ["IBM", "IONQ", "RGTI"],      # Quantinuum — quantum computing
        "STUB": ["LYV", "SEAS", "BKNG"],      # StubHub — ticketing
        "MDLN": ["ABT", "MDT", "BDX"],        # Medline — medical supplies
        # ── US Mega-cap Tech ──────────────────────────────────────────────
        "MSFT": ["AAPL", "GOOGL", "AMZN"],
        "AAPL": ["MSFT", "GOOGL", "SAMSUNG"],
        "GOOGL": ["MSFT", "META", "AMZN"],
        "GOOG":  ["MSFT", "META", "AMZN"],
        "AMZN": ["MSFT", "GOOGL", "BABA"],
        "META": ["GOOGL", "SNAP", "PINS"],
        "NVDA": ["AMD", "INTC", "AVGO"],
        "AMD":  ["NVDA", "INTC", "QCOM"],
        "INTC": ["AMD", "NVDA", "QCOM"],
        "AVGO": ["NVDA", "QCOM", "MRVL"],
        "QCOM": ["AVGO", "AMD", "MRVL"],
        "TSM":  ["INTC", "SSNLF", "ASML"],
        "ASML": ["AMAT", "LRCX", "KLAC"],
        "AMAT": ["LRCX", "KLAC", "ASML"],
        "ARM":  ["NVDA", "INTC", "QCOM"],
        # ── US Enterprise / SaaS ─────────────────────────────────────────
        "CRM":  ["NOW", "ORCL", "WDAY"],
        "NOW":  ["CRM", "WDAY", "SAP"],
        "WDAY": ["CRM", "NOW", "ORCL"],
        "ORCL": ["SAP", "MSFT", "NOW"],
        "SAP":  ["ORCL", "CRM", "NOW"],
        "ADBE": ["CRM", "MSFT", "FIGMA"],
        "SNOW": ["DDOG", "MDB", "NET"],
        "DDOG": ["SNOW", "SPLK", "NET"],
        "MDB":  ["SNOW", "DDOG", "ESTC"],
        "NET":  ["FSLY", "DDOG", "CRWD"],
        "CRWD": ["PANW", "FTNT", "S"],
        "PANW": ["CRWD", "FTNT", "CHKP"],
        "HUBS": ["CRM", "NOW", "MKTO"],
        "SHOP": ["AMZN", "WMT", "BIGC"],
        "PLTR": ["SAIC", "LEIDOS", "BAH"],
        # ── US Internet / Consumer Tech ───────────────────────────────────
        "NFLX": ["DIS", "WBD", "PARA"],
        "DIS":  ["NFLX", "WBD", "PARA"],
        "TSLA": ["GM", "F", "RIVN"],
        "RIVN": ["TSLA", "LCID", "NIO"],
        "LCID": ["TSLA", "RIVN", "NIO"],
        "UBER": ["LYFT", "DASH", "GRAB"],
        "LYFT": ["UBER", "DASH", "GRAB"],
        "ABNB": ["BKNG", "EXPE", "TRIP"],
        "BKNG": ["ABNB", "EXPE", "TRIP"],
        "RBLX": ["U", "EA", "TTWO"],
        # ── US Finance ───────────────────────────────────────────────────
        "JPM":  ["BAC", "GS", "MS"],
        "BAC":  ["JPM", "WFC", "C"],
        "GS":   ["MS", "JPM", "C"],
        "MS":   ["GS", "JPM", "BLK"],
        "WFC":  ["JPM", "BAC", "USB"],
        "C":    ["JPM", "BAC", "WFC"],
        "BLK":  ["TROW", "BEN", "IVZ"],
        "V":    ["MA", "AXP", "PYPL"],
        "MA":   ["V", "AXP", "PYPL"],
        "AXP":  ["V", "MA", "DFS"],
        "PYPL": ["SQ", "V", "MA"],
        "SQ":   ["PYPL", "V", "AFRM"],
        # ── US Healthcare / Pharma ────────────────────────────────────────
        "JNJ":  ["PFE", "MRK", "ABT"],
        "PFE":  ["MRK", "JNJ", "ABBV"],
        "MRK":  ["PFE", "JNJ", "BMY"],
        "ABBV": ["PFE", "MRK", "BMY"],
        "LLY":  ["NVO", "PFE", "ABBV"],
        "NVO":  ["LLY", "PFE", "AZN"],
        "UNH":  ["CVS", "CI", "HUM"],
        # ── US Consumer / Retail ──────────────────────────────────────────
        "WMT":  ["TGT", "COST", "AMZN"],
        "TGT":  ["WMT", "COST", "KR"],
        "COST": ["WMT", "TGT", "BJ"],
        "MCD":  ["SBUX", "QSR", "YUM"],
        "SBUX": ["MCD", "DNKN", "QSR"],
        "NKE":  ["ADDYY", "PUMA", "UAA"],
        "KO":   ["PEP", "MNST", "KDP"],
        "PEP":  ["KO", "MNST", "KDP"],
        # ── US Energy / Industrial ────────────────────────────────────────
        "XOM":  ["CVX", "COP", "BP"],
        "CVX":  ["XOM", "COP", "SHEL"],
        "BA":   ["LMT", "RTX", "NOC"],
        "LMT":  ["BA", "RTX", "NOC"],
        "RTX":  ["LMT", "BA", "NOC"],
        "CAT":  ["DE", "CMI", "PCAR"],
        # ── Auto (Global) ─────────────────────────────────────────────────
        "TM":   ["NSANY", "HMC", "HYMTF"],      # Toyota
        "NSANY":["TM", "HMC", "HYMTF"],          # Nissan ADR
        "HMC":  ["TM", "NSANY", "HYMTF"],        # Honda ADR
        "HYMTF":["TM", "HMC", "VWAGY"],          # Hyundai
        "KIMTF":["HYMTF", "TM", "HMC"],          # Kia
        "VWAGY":["BMWYY", "MBGYY", "STLA"],      # VW
        "BMWYY":["VWAGY", "MBGYY", "STLA"],      # BMW
        "MBGYY":["BMWYY", "VWAGY", "STLA"],      # Mercedes
        "STLA": ["GM", "F", "VWAGY"],             # Stellantis
        "GM":   ["F", "STLA", "TSLA"],
        "F":    ["GM", "STLA", "TSLA"],
        "FUJHY":["TM", "HMC", "NSANY"],           # Subaru
        "MZDAY":["TM", "HMC", "NSANY"],           # Mazda
        "NIO":  ["TSLA", "RIVN", "XPEV"],
        "XPEV": ["NIO", "TSLA", "LI"],
        "LI":   ["NIO", "XPEV", "TSLA"],
        "BYDDY":["TSLA", "NIO", "HYMTF"],         # BYD
        "TTM":  ["HMC", "TM", "NSANY"],           # Tata Motors
        # ── Japan (other) ─────────────────────────────────────────────────
        "SONY": ["NTDOY", "SEGA", "EA"],
        "NTDOY":["SONY", "EA", "TTWO"],
        "SFTBY":["TCEHY", "BABA", "META"],
        # ── Europe ────────────────────────────────────────────────────────
        "SHEL": ["BP", "XOM", "TTE"],
        "BP":   ["SHEL", "XOM", "TTE"],
        "TTE":  ["SHEL", "BP", "ENI"],
        "HSBC": ["BCS", "DB", "UBS"],
        "BCS":  ["HSBC", "LYG", "DB"],
        "UL":   ["NSRGY", "PG", "CL"],
        "NSRGY":["UL", "PG", "KHC"],
        "AZN":  ["NVO", "GSK", "SNY"],
        "GSK":  ["AZN", "SNY", "NVS"],
        "NVS":  ["RHHBY", "AZN", "GSK"],
        "RHHBY":["NVS", "AZN", "ABBV"],
        "SIEGY":["ABB", "PHG", "HON"],
        "LVMUY":["PPRUY", "CFRUY", "BURBY"],
        "EADSY":["BA", "RTX", "LMT"],
        "RYCEY":["GE", "RTX", "SAFRY"],
        "SPOT": ["AAPL", "AMZN", "TIDAL"],
        "ARM":  ["NVDA", "INTC", "QCOM"],
        # ── India ─────────────────────────────────────────────────────────
        "INFY": ["WIT", "TCS.NS", "HCLTECH.NS"],
        "WIT":  ["INFY", "TCS.NS", "TECHM.NS"],
        "TCS.NS":   ["INFY", "WIT", "HCLTECH.NS"],
        "HCLTECH.NS":["TCS.NS", "INFY", "TECHM.NS"],
        "IBN":  ["HDFCBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
        "HDFCBANK.NS":["IBN", "KOTAKBANK.NS", "AXISBANK.NS"],
        "RELIANCE.NS":["ADANIENT.NS", "TCS.NS", "HDFCBANK.NS"],
        # ── China ─────────────────────────────────────────────────────────
        "BABA": ["JD", "PDD", "TCEHY"],
        "TCEHY":["BABA", "BIDU", "NTES"],
        "BIDU": ["GOOGL", "TCEHY", "BABA"],
        "JD":   ["BABA", "PDD", "AMZN"],
        "PDD":  ["BABA", "JD", "AMZN"],
        # ── Semiconductors ────────────────────────────────────────────────
        "SSNLF":["TSM", "INTC", "NVDA"],
        "MU":   ["WDC", "SSNLF", "STX"],
        "WDC":  ["MU", "STX", "NTAP"],
    }

    result = peer_map.get(ticker.upper(), [])

    # Fallback: use sector to suggest generic peers if no map entry
    if not result:
        sector = info.get("sector", "")
        sector_defaults = {
            "Technology": ["MSFT", "AAPL", "GOOGL"],
            "Consumer Cyclical": ["AMZN", "WMT", "TGT"],
            "Financial Services": ["JPM", "BAC", "GS"],
            "Healthcare": ["JNJ", "PFE", "UNH"],
            "Communication Services": ["GOOGL", "META", "NFLX"],
            "Energy": ["XOM", "CVX", "SHEL"],
            "Industrials": ["GE", "CAT", "BA"],
            "Basic Materials": ["BHP", "RIO", "FCX"],
            "Consumer Defensive": ["PG", "KO", "WMT"],
            "Real Estate": ["AMT", "PLD", "EQIX"],
            "Utilities": ["NEE", "DUK", "SO"],
        }
        result = sector_defaults.get(sector, [])
        # Remove current ticker from suggestions if present
        result = [t for t in result if t != ticker.upper()][:3]

    return result
