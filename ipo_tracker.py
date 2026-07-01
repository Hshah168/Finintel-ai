import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


# ─── Curated IPO database ─────────────────────────────────────────────────────
# (ticker, name, type, ipo_date, ipo_price, exchange)
IPO_DATABASE = [
    # ── 2026 ──────────────────────────────────────────────────────────────────
    ("SPCX",  "SpaceX",                          "STOCK", "2026-06-12", 135.00, "NASDAQ"),
    ("CBRS",  "Cerebras Systems",                "STOCK", "2026-05-14",  37.00, "NASDAQ"),
    ("QNT",   "Quantinuum",                      "STOCK", "2026-06-04",  22.00, "NASDAQ"),
    ("ALAB",  "Astera Labs",                     "STOCK", "2026-03-20",  36.00, "NASDAQ"),
    ("RZLV",  "Rezolve AI",                      "STOCK", "2026-01-30",  12.00, "NASDAQ"),

    # ── 2025 ──────────────────────────────────────────────────────────────────
    ("CRWV",  "CoreWeave",                       "STOCK", "2025-03-28",  40.00, "NASDAQ"),
    ("CRCL",  "Circle Internet Group",           "STOCK", "2025-06-05",  31.00, "NYSE"),
    ("STUB",  "StubHub",                         "STOCK", "2025-09-18",  28.00, "NYSE"),
    ("MDLN",  "Medline Industries",              "STOCK", "2025-12-17",  23.00, "NASDAQ"),
    ("FIG",   "Figma",                           "STOCK", "2025-10-02",  42.00, "NYSE"),
    ("RBRK",  "Rubrik",                          "STOCK", "2024-04-25",  32.00, "NYSE"),

    # ── 2024 ──────────────────────────────────────────────────────────────────
    ("RDDT",  "Reddit",                          "STOCK", "2024-03-21",  34.00, "NYSE"),
    ("IBIT",  "iShares Bitcoin Trust ETF",       "ETF",   "2024-01-11",  26.00, "NASDAQ"),
    ("FBTC",  "Fidelity Wise Origin Bitcoin ETF","ETF",   "2024-01-11",  37.00, "CBOE"),
    ("BITB",  "Bitwise Bitcoin ETF",             "ETF",   "2024-01-11",  23.00, "NYSE"),
    ("DJT",   "Trump Media & Technology",        "STOCK", "2024-03-26",  49.95, "NASDAQ"),
    ("SERV",  "Serve Robotics",                  "STOCK", "2024-04-18",   4.00, "NASDAQ"),

    # ── 2023 ──────────────────────────────────────────────────────────────────
    ("ARM",   "Arm Holdings",                    "STOCK", "2023-09-14",  51.00, "NASDAQ"),
    ("KVYO",  "Klaviyo",                         "STOCK", "2023-09-20",  30.00, "NYSE"),
    ("BIRK",  "Birkenstock",                     "STOCK", "2023-10-11",  46.00, "NYSE"),
    ("CART",  "Instacart (Maplebear)",           "STOCK", "2023-09-19",  30.00, "NASDAQ"),
    ("LUNR",  "Intuitive Machines",              "STOCK", "2023-02-14",  10.00, "NASDAQ"),

    # ── ETF Launches (2023-2025) ───────────────────────────────────────────────
    ("NVDY",  "YieldMax NVDA Option ETF",        "ETF",   "2023-09-01",  20.00, "NYSE"),
    ("MSFO",  "YieldMax MSFT Option ETF",        "ETF",   "2023-09-01",  20.00, "NYSE"),
    ("CONY",  "YieldMax COIN Option ETF",        "ETF",   "2023-08-14",  20.00, "NYSE"),
    ("WTAI",  "WisdomTree AI & Innovation ETF",  "ETF",   "2023-10-25",  33.00, "NASDAQ"),
    ("AIAI",  "Global X AI & Technology ETF",    "ETF",   "2025-03-15",  25.00, "NASDAQ"),
    ("AIXI",  "Xoshido AI Chips ETF",            "ETF",   "2025-01-10",  25.00, "NASDAQ"),

    # ── Space & Deep Tech (recent listings) ───────────────────────────────────
    ("ASTS",  "AST SpaceMobile",                 "STOCK", "2021-04-07",  10.00, "NASDAQ"),
    ("RKLB",  "Rocket Lab USA",                  "STOCK", "2021-08-25",  10.00, "NASDAQ"),
    ("IONQ",  "IonQ",                            "STOCK", "2021-10-01",  10.00, "NYSE"),
    ("RGTI",  "Rigetti Computing",               "STOCK", "2022-03-02",  10.00, "NASDAQ"),
    ("ACHR",  "Archer Aviation",                 "STOCK", "2021-09-17",  10.00, "NYSE"),
    ("JOBY",  "Joby Aviation",                   "STOCK", "2021-08-10",  10.00, "NYSE"),
    ("BBAI",  "BigBear.ai",                      "STOCK", "2021-12-08",  10.00, "NYSE"),
]


def _days_since_ipo(ipo_date_str: str) -> int:
    try:
        return (datetime.now() - datetime.strptime(ipo_date_str, "%Y-%m-%d")).days
    except Exception:
        return 0


@st.cache_data(ttl=1800)
def fetch_ipo_performance(period: str = "Last 30 Days") -> pd.DataFrame:
    """
    Fetch current prices and calculate performance since IPO listing.
    Returns DataFrame with all columns needed for display.
    """
    days_limit = 7 if period == "Last 7 Days" else 30
    cutoff = datetime.now() - timedelta(days=days_limit)

    recent = [
        {"ticker": r[0], "name": r[1], "type": r[2],
         "ipo_date": r[3], "ipo_price": r[4], "exchange": r[5]}
        for r in IPO_DATABASE
        if datetime.strptime(r[3], "%Y-%m-%d") >= cutoff
    ]

    # If window is too narrow, show most recent 20 from full database
    if not recent:
        sorted_db = sorted(IPO_DATABASE, key=lambda x: x[3], reverse=True)
        recent = [
            {"ticker": r[0], "name": r[1], "type": r[2],
             "ipo_date": r[3], "ipo_price": r[4], "exchange": r[5]}
            for r in sorted_db[:20]
        ]

    rows = []
    for item in recent:
        ticker = item["ticker"]
        ipo_price = item["ipo_price"]
        row = {
            "ticker":      ticker,
            "name":        item["name"],
            "type":        item["type"],
            "ipo_date":    item["ipo_date"],
            "exchange":    item["exchange"],
            "ipo_price":   ipo_price,
            "curr_price":  None,
            "change_pct":  None,
            "change_dol":  None,
            "days_listed": _days_since_ipo(item["ipo_date"]),
        }
        try:
            info = yf.Ticker(ticker).info
            curr = (info.get("currentPrice") or
                    info.get("regularMarketPrice") or
                    info.get("previousClose"))
            if curr:
                curr = float(curr)
                row["curr_price"] = curr
                row["change_dol"] = curr - ipo_price
                row["change_pct"] = (curr - ipo_price) / ipo_price * 100
        except Exception:
            pass
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("ipo_date", ascending=False).reset_index(drop=True)
    return df


def get_ipo_stats(df: pd.DataFrame) -> dict:
    """Summary statistics for the IPO tracker header cards."""
    if df is None or df.empty:
        return {}
    priced = df[df["change_pct"].notna()]
    return {
        "total":        len(df),
        "with_price":   len(priced),
        "gainers":      int((priced["change_pct"] > 0).sum()),
        "losers":       int((priced["change_pct"] < 0).sum()),
        "avg_return":   float(priced["change_pct"].mean()) if not priced.empty else 0,
        "best_ticker":  priced.loc[priced["change_pct"].idxmax(), "ticker"] if not priced.empty else "N/A",
        "best_return":  float(priced["change_pct"].max()) if not priced.empty else 0,
        "worst_ticker": priced.loc[priced["change_pct"].idxmin(), "ticker"] if not priced.empty else "N/A",
        "worst_return": float(priced["change_pct"].min()) if not priced.empty else 0,
    }
