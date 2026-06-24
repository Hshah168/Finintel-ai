"""
news.py
-------
Fetches financial news using multiple sources with fallbacks.
yfinance >= 0.2.37 changed the news response structure —
articles now nest content under a 'content' key.
"""

import yfinance as yf
import requests
import streamlit as st
from datetime import datetime


def _parse_yf_article(item: dict) -> dict | None:
    """
    Handle both old and new yfinance news response formats.

    Old format (< 0.2.37):  item has 'title', 'publisher', 'link', 'providerPublishTime'
    New format (>= 0.2.37): item has 'content' dict with 'title', 'provider', 'canonicalUrl', 'pubDate'
    """
    # --- New format ---
    content = item.get("content", {})
    if content and content.get("title"):
        title = content.get("title", "").strip()
        if not title or title == "No title":
            return None

        provider = content.get("provider", {})
        publisher = provider.get("displayName") or provider.get("name") or "Unknown"

        canonical = content.get("canonicalUrl", {})
        link = canonical.get("url") or content.get("clickThroughUrl", {}).get("url") or "#"

        pub_date_raw = content.get("pubDate", "")
        try:
            dt = datetime.fromisoformat(pub_date_raw.replace("Z", "+00:00"))
            pub_date = dt.strftime("%b %d, %Y %H:%M")
            timestamp = dt.timestamp()
        except Exception:
            pub_date = pub_date_raw[:10] if pub_date_raw else "Unknown"
            timestamp = 0

        return {
            "title": title,
            "publisher": publisher,
            "link": link,
            "published_at": pub_date,
            "timestamp": timestamp,
        }

    # --- Old format ---
    title = item.get("title", "").strip()
    if not title:
        return None

    pub_ts = item.get("providerPublishTime", 0)
    pub_date = (
        datetime.fromtimestamp(pub_ts).strftime("%b %d, %Y %H:%M")
        if pub_ts else "Unknown"
    )

    return {
        "title": title,
        "publisher": item.get("publisher", "Unknown"),
        "link": item.get("link", "#"),
        "published_at": pub_date,
        "timestamp": pub_ts,
    }


@st.cache_data(ttl=900)
def get_company_news(ticker: str, company_name: str, limit: int = 12) -> list[dict]:
    """
    Fetch latest news articles for a company.
    Returns list of dicts sorted newest first.
    """
    articles = []

    # Primary: yfinance (handles both old and new response format)
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news or []

        # yfinance may also expose get_news()
        if not raw_news:
            try:
                raw_news = t.get_news() or []
            except Exception:
                pass

        for item in raw_news:
            parsed = _parse_yf_article(item)
            if parsed:
                articles.append(parsed)
    except Exception:
        pass

    # Fallback: Yahoo Finance REST search API (news endpoint)
    if len(articles) < 3:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            url = "https://query1.finance.yahoo.com/v1/finance/search"
            params = {
                "q": company_name,
                "newsCount": limit,
                "quotesCount": 0,
                "enableFuzzyQuery": False,
            }
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("news", []):
                    parsed = _parse_yf_article(item)
                    if parsed:
                        articles.append(parsed)
        except Exception:
            pass

    # Second fallback: Yahoo Finance v2 news API
    if len(articles) < 3:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            url = f"https://query2.finance.yahoo.com/v1/finance/search"
            params = {"q": ticker, "newsCount": limit, "quotesCount": 0}
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            if resp.status_code == 200:
                for item in resp.json().get("news", []):
                    parsed = _parse_yf_article(item)
                    if parsed and parsed["title"] not in {a["title"] for a in articles}:
                        articles.append(parsed)
        except Exception:
            pass

    # Deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)

    # Sort newest first
    unique.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return unique[:limit]
