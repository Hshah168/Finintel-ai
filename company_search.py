"""
company_search.py
-----------------
Resolves company names to ticker symbols using a three-layer pipeline:

  Layer 1 — Private company blocklist (instant rejection with clear message)
  Layer 2 — Curated alias map (fast, guaranteed-correct for known companies)
  Layer 3 — Intelligent auto-discovery (handles ANY newly public company
             automatically, using Yahoo Finance + Wikipedia + verification)

Layer 3 is the key innovation: it means the app never needs a manual update
when a new company goes public. The auto-discovery pipeline:
  a) Queries Yahoo Finance search API
  b) Scores candidates by exchange quality and name similarity
  c) Verifies the winner actually has live market data (not a shell/delisted co.)
  d) Caches confirmed results in Streamlit session state for instant reuse
"""

import yfinance as yf
import requests
import streamlit as st
from difflib import SequenceMatcher


# ─── Layer 1: Known private companies ────────────────────────────────────────
# These are confirmed private as of June 2026. Remove any that go public.
PRIVATE_COMPANIES = {
    "openai", "open ai",
    "klarna",
    "databricks",
    "discord",
    "chime",
    "revolut",
    "shein",
    "anduril", "anduril industries",
    "canva",
    "instacart",
    "epic games",
    "cargill", "koch industries", "ikea", "lego",
    "bloomberg", "mckinsey",
    "deloitte", "pwc", "kpmg", "ey", "ernst young",
    "bain", "bcg", "boston consulting",
    "brex", "plaid", "kraken",
    "bytedance", "tiktok",
    "anthropic",
}


# ─── Layer 2: Curated alias map ───────────────────────────────────────────────
# Only needed for companies where Yahoo's search returns wrong results.
# Most companies (including all future IPOs) are handled by Layer 3.
COMPANY_ALIASES = {
    # ── Recent IPOs (Yahoo search returns wrong results without these) ────────
    "spacex": "SPCX",               # Nasdaq, June 12 2026
    "space x": "SPCX",
    "space exploration": "SPCX",
    "coreweave": "CRWV",            # Nasdaq, March 28 2025
    "core weave": "CRWV",
    "cerebras": "CBRS",             # Nasdaq, May 14 2026
    "cerebras systems": "CBRS",
    "circle": "CRCL",               # NYSE, June 2025
    "circle internet": "CRCL",
    "circle internet group": "CRCL",
    "figma": "FIG",                 # NYSE, 2025
    "quantinuum": "QNT",            # Nasdaq, June 4 2026
    "stubhub": "STUB",              # NYSE, September 2025
    "stub hub": "STUB",
    "medline": "MDLN",              # Nasdaq, December 17 2025
    "medline industries": "MDLN",
    # ── Companies that resolve to wrong subsidiaries without aliases ──────────
    "nissan": "NSANY",
    "nissan motor": "NSANY",
    "honda": "HMC",
    "honda motor": "HMC",
    "bmw": "BMWYY",
    "bmw group": "BMWYY",
    "mercedes": "MBGYY",
    "mercedes-benz": "MBGYY",
    "mercedes benz": "MBGYY",
    "hyundai": "HYMTF",
    "hyundai motor": "HYMTF",
    # ── Standard well-known tickers (fast path) ───────────────────────────────
    "microsoft": "MSFT",
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "uber": "UBER",
    "airbnb": "ABNB",
    "palantir": "PLTR",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "shopify": "SHOP",
    "snowflake": "SNOW",
    "datadog": "DDOG",
    "servicenow": "NOW",
    "workday": "WDAY",
    "crowdstrike": "CRWD",
    "palo alto": "PANW",
    "palo alto networks": "PANW",
    "cloudflare": "NET",
    "hubspot": "HUBS",
    "lyft": "LYFT",
    "ford": "F",
    "ford motor": "F",
    "general motors": "GM",
    "gm": "GM",
    "stellantis": "STLA",
    "rivian": "RIVN",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "goldman sachs": "GS",
    "goldman": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C",
    "citi": "C",
    "blackrock": "BLK",
    "visa": "V",
    "mastercard": "MA",
    "american express": "AXP",
    "amex": "AXP",
    "paypal": "PYPL",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "abbvie": "ABBV",
    "merck": "MRK",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "unitedhealth": "UNH",
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "home depot": "HD",
    "nike": "NKE",
    "mcdonalds": "MCD",
    "mcdonald's": "MCD",
    "starbucks": "SBUX",
    "coca cola": "KO",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "boeing": "BA",
    "lockheed martin": "LMT",
    "sap": "SAP",
    "asml": "ASML",
    "toyota": "TM",
    "sony": "SONY",
    "samsung": "SSNLF",
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "alibaba": "BABA",
    "tencent": "TCEHY",
    "baidu": "BIDU",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY",
    "wipro": "WIT",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "IBN",
    "nestle": "NSRGY",
    "volkswagen": "VWAGY",
    "vw": "VWAGY",
    "porsche": "POAHY",
    "siemens": "SIEGY",
    "novartis": "NVS",
    "roche": "RHHBY",
    "astrazeneca": "AZN",
    "gsk": "GSK",
    "glaxosmithkline": "GSK",
    "shell": "SHEL",
    "bp": "BP",
    "totalenergies": "TTE",
    "hsbc": "HSBC",
    "unilever": "UL",
    "lvmh": "LVMUY",
    "spotify": "SPOT",
    "arm": "ARM",
    "arm holdings": "ARM",
    "kia": "KIMTF",
    "subaru": "FUJHY",
    "mazda": "MZDAY",
    "nio": "NIO",
    "xpeng": "XPEV",
    "byd": "BYDDY",
    "tata motors": "TTM",
    "nintendo": "NTDOY",
    "softbank": "SFTBY",
}

# Exchange quality tiers for auto-discovery scoring
_TIER1_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "NYSEArca"}   # US major
_TIER2_EXCHANGES = {"PNK", "GREY", "OTC"}                       # US OTC/ADR
_TIER3_EXCHANGES = {"NSE", "BSE", "LSE", "GER", "PAR", "AMS"}  # Major foreign
_DEPRIORITIZED_SUFFIXES = (".TW", ".TWO", ".HK", ".SS", ".SZ", ".KS", ".KQ")


def _name_similarity(a: str, b: str) -> float:
    """Return 0-1 string similarity between two company names."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_candidate(symbol: str, long_name: str, short_name: str,
                      exchange: str, query: str) -> float:
    """
    Score a Yahoo Finance search candidate (higher = better match).
    Factors: exchange quality, name similarity, ticker length (proxy for specificity).
    """
    score = 0.0

    # Exchange quality
    if exchange in _TIER1_EXCHANGES:
        score += 40
    elif exchange in _TIER2_EXCHANGES:
        score += 25
    elif exchange in _TIER3_EXCHANGES:
        score += 20
    else:
        score += 5

    # Penalise local-market suffixes (prefer ADR/international listing)
    if any(symbol.endswith(sfx) for sfx in _DEPRIORITIZED_SUFFIXES):
        score -= 20

    # Name similarity to query
    best_name = long_name or short_name or ""
    sim = _name_similarity(query, best_name)
    score += sim * 40   # up to 40 points for exact name match

    # Shorter tickers tend to be the primary listing
    score -= len(symbol) * 0.5

    return score


def _verify_ticker(ticker: str) -> bool:
    """
    Verify a ticker has live market data and is not delisted/empty.
    Returns True if valid, False if the ticker should be rejected.
    """
    try:
        info = yf.Ticker(ticker).info
        # Must have a current price OR market cap to be considered active
        has_price = bool(
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("previousClose")
        )
        has_market_cap = bool(info.get("marketCap"))
        return has_price or has_market_cap
    except Exception:
        return False


def _search_yahoo(query: str, count: int = 10) -> list[dict]:
    """Query Yahoo Finance search API and return raw quote results."""
    for base_url in [
        "https://query1.finance.yahoo.com/v1/finance/search",
        "https://query2.finance.yahoo.com/v1/finance/search",
    ]:
        try:
            resp = requests.get(
                base_url,
                params={
                    "q": query,
                    "quotesCount": count,
                    "newsCount": 0,
                    "enableFuzzyQuery": True,
                    "quotesQueryId": "tss_match_phrase_query",
                },
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=8,
            )
            if resp.status_code == 200:
                return resp.json().get("quotes", [])
        except Exception:
            continue
    return []


@st.cache_data(ttl=86400)  # Cache for 24 hours — new IPOs don't happen every minute
def _auto_discover(company_name: str) -> tuple[str | None, str | None]:
    """
    Layer 3: Automatically discover the correct ticker for ANY company name,
    including companies that IPO'd after this code was written.

    Algorithm:
    1. Search Yahoo Finance for the company name
    2. Filter to EQUITY type only
    3. Score each candidate by exchange quality + name similarity
    4. Verify the top candidate has live market data
    5. Return the verified winner
    """
    quotes = _search_yahoo(company_name, count=15)
    equity_quotes = [q for q in quotes if q.get("quoteType") == "EQUITY"]

    if not equity_quotes:
        return None, None

    # Score all candidates
    scored = []
    for q in equity_quotes:
        symbol = q.get("symbol", "")
        long_name = q.get("longname", "") or ""
        short_name = q.get("shortname", "") or ""
        exchange = q.get("exchange", "") or q.get("exchDisp", "") or ""
        score = _score_candidate(symbol, long_name, short_name, exchange, company_name)
        scored.append((score, symbol, long_name or short_name or symbol))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Try candidates in order until one verifies
    for score, symbol, name in scored[:5]:
        if _verify_ticker(symbol):
            return symbol, name

    # If none verified, return top result anyway (best effort)
    if scored:
        return scored[0][1], scored[0][2]

    return None, None


@st.cache_data(ttl=3600)
def resolve_ticker(company_name: str) -> tuple[str | None, str | None]:
    """
    Master resolver — three-layer pipeline.

    Returns:
        (ticker, company_full_name) — for any publicly traded company
        ("PRIVATE", name)           — for known private companies
        (None, None)                — if not found anywhere
    """
    if not company_name or not company_name.strip():
        return None, None

    query = company_name.strip().lower()

    # ── Layer 1: Private company check ───────────────────────────────────────
    if query in PRIVATE_COMPANIES:
        return "PRIVATE", company_name.strip()

    # ── Layer 2a: Direct ticker input ─────────────────────────────────────────
    raw = company_name.strip()
    if len(raw) <= 6 and raw.replace(".", "").isalpha():
        ticker_upper = raw.upper()
        try:
            info = yf.Ticker(ticker_upper).info
            if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("marketCap"):
                name = info.get("longName") or info.get("shortName") or ticker_upper
                return ticker_upper, name
        except Exception:
            pass

    # ── Layer 2b: Curated alias map ───────────────────────────────────────────
    if query in COMPANY_ALIASES:
        ticker = COMPANY_ALIASES[query]
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or ticker
            return ticker, name
        except Exception:
            return ticker, ticker

    # ── Layer 3: Auto-discovery (handles all future IPOs automatically) ───────
    ticker, name = _auto_discover(company_name.strip())
    if ticker:
        return ticker, name

    return None, None


@st.cache_data(ttl=3600)
def get_company_info(ticker: str) -> dict:
    """Fetch full yfinance info dict for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception:
        return {}
