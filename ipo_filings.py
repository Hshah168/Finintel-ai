"""SEC EDGAR-backed IPO registration filing tracker.

The feed is refreshed every 30 minutes. Registration filings are leads for
research, not confirmation that an IPO will price or list.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from html import escape, unescape

import pandas as pd
import requests
import streamlit as st

SEC_FEED_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_FORMS = ("S-1", "S-1/A", "F-1", "F-1/A")
SEC_HEADERS = {
    "User-Agent": "FinIntel AI research app (https://github.com/Hshah168/Finintel-ai)",
    "Accept-Encoding": "gzip, deflate",
}
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _plain_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _filing_form(title: str, fallback: str) -> str:
    match = re.search(r"\b(S-1/A|S-1|F-1/A|F-1)\b", title or "")
    return match.group(1) if match else fallback


def _filing_date(summary: str, updated: str) -> str:
    match = re.search(r"(?:Filed|Filing Date)\s*:?\s*(\d{4}-\d{2}-\d{2})", summary or "", re.I)
    if match:
        return match.group(1)
    if updated:
        try:
            return datetime.fromisoformat(updated.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_recent_ipo_filings(days: int = 90, limit: int = 100) -> pd.DataFrame:
    """Fetch current S-1/F-1 registration filings from SEC EDGAR Atom feeds."""
    cutoff = date.today() - timedelta(days=max(1, int(days)))
    records: list[dict] = []

    for form in SEC_FORMS:
        try:
            response = requests.get(
                SEC_FEED_URL,
                params={
                    "action": "getcurrent",
                    "CIK": "",
                    "type": form,
                    "company": "",
                    "dateb": "",
                    "owner": "include",
                    "start": 0,
                    "count": min(max(int(limit), 10), 100),
                    "output": "atom",
                },
                headers=SEC_HEADERS,
                timeout=20,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for entry in root.findall("atom:entry", ATOM_NS):
                title_el = entry.find("atom:title", ATOM_NS)
                summary_el = entry.find("atom:summary", ATOM_NS)
                updated_el = entry.find("atom:updated", ATOM_NS)
                title = _plain_text(title_el.text if title_el is not None else "")
                summary = "".join(summary_el.itertext()) if summary_el is not None else ""
                updated = updated_el.text if updated_el is not None else ""

                links = entry.findall("atom:link", ATOM_NS)
                filing_url = next(
                    (link.attrib.get("href", "") for link in links
                     if link.attrib.get("rel") == "alternate" and link.attrib.get("href")),
                    next((link.attrib.get("href", "") for link in links if link.attrib.get("href")), ""),
                )
                form_found = _filing_form(title, form)
                filed = _filing_date(summary, updated)
                if filed:
                    try:
                        if date.fromisoformat(filed) < cutoff:
                            continue
                    except ValueError:
                        pass

                company = re.sub(
                    r"\s*\((?:S-1/A|S-1|F-1/A|F-1)\)\s*$",
                    "", title, flags=re.I,
                ).strip() or title or "Company name unavailable"
                records.append({
                    "Company": company,
                    "Form": form_found,
                    "Filed": filed or "Date unavailable",
                    "Filing": filing_url,
                    "Summary": _plain_text(summary),
                    "Updated": updated,
                })
        except (requests.RequestException, ET.ParseError, ValueError):
            # A failed form feed should not prevent results from other forms.
            continue

    columns = ["Company", "Form", "Filed", "Filing", "Summary", "Updated"]
    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records, columns=columns)
    df = df.drop_duplicates(subset=["Filing", "Company", "Form"])
    df["_sort_date"] = pd.to_datetime(df["Filed"], errors="coerce")
    df = df.sort_values(["_sort_date", "Company"], ascending=[False, True], na_position="last")
    return df.drop(columns=["_sort_date"]).head(max(1, int(limit))).reset_index(drop=True)


def render_ipo_tracker(compact: bool = False) -> None:
    """Render a landing-page preview or the full IPO filing tracker."""
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">'
        '<div><p style="color:#0A84FF;font-size:11px;font-weight:700;letter-spacing:1px;'
        'text-transform:uppercase;margin:0 0 5px">IPO intelligence</p>'
        '<h2 style="color:#FFFFFF;margin:0;font-size:24px">IPO Filing & Opportunity Tracker</h2>'
        '<p style="color:#8E8E93;font-size:13px;margin:7px 0 0">Discover companies entering the public-market pipeline from SEC registration filings.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if compact:
        days, limit, form_filter = 90, 5, list(SEC_FORMS)
    else:
        filter_cols = st.columns([1, 1, 2])
        with filter_cols[0]:
            days = st.selectbox("Look back", [30, 60, 90, 180], index=2,
                                format_func=lambda d: f"Last {d} days")
        with filter_cols[1]:
            limit = st.selectbox("Results", [10, 25, 50, 100], index=2)
        with filter_cols[2]:
            form_filter = st.multiselect(
                "Registration form", list(SEC_FORMS),
                default=["S-1", "S-1/A", "F-1", "F-1/A"],
            )

    with st.spinner("Checking SEC EDGAR registration filings…"):
        filings = fetch_recent_ipo_filings(days=days, limit=100 if compact else limit)

    if filings.empty:
        st.info(
            "SEC EDGAR did not return registration filings right now. "
            "Try again shortly, or search the official SEC filing database."
        )
        st.markdown("[Open SEC EDGAR filing search](https://www.sec.gov/edgar/search/)")
        return

    if not compact:
        filings = filings[filings["Form"].isin(form_filter)] if form_filter else filings.iloc[0:0]
        search = st.text_input("Search company names", placeholder="Filter by company name…")
        if search.strip():
            filings = filings[filings["Company"].str.contains(search.strip(), case=False, na=False)]

    total = len(filings)
    s1_count = int(filings["Form"].isin(["S-1", "S-1/A"]).sum())
    f1_count = int(filings["Form"].isin(["F-1", "F-1/A"]).sum())
    metric_cols = st.columns(3)
    metric_cols[0].metric("Registration filings", total)
    metric_cols[1].metric("U.S. issuer forms", s1_count)
    metric_cols[2].metric("Foreign issuer forms", f1_count)

    if not compact:
        st.caption("Source: SEC EDGAR current filings · Cached for 30 minutes · Counts represent filings, not unique IPOs.")

    if total == 0:
        st.info("No filings match those filters. Try a longer look-back window or different form types.")
        return

    for _, row in filings.head(5 if compact else limit).iterrows():
        form = str(row.get("Form", ""))
        form_label = "Foreign issuer" if form.startswith("F-1") else "U.S. issuer"
        filed = str(row.get("Filed", "Date unavailable"))
        summary = str(row.get("Summary", ""))
        url = str(row.get("Filing", ""))
        company_html = escape(str(row["Company"]))
        summary_html = escape(summary[:240])
        url_html = (
            f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#0A84FF;font-size:12px;text-decoration:none">View SEC filing ↗</a>'
            if url.startswith("https://www.sec.gov/") else ""
        )
        st.markdown(
            f'<div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:12px;'
            f'padding:15px 18px;margin:10px 0">'
            f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">'
            f'<div><p style="color:#FFFFFF;font-size:14px;font-weight:700;margin:0 0 5px">{company_html}</p>'
            f'<p style="color:#8E8E93;font-size:11px;margin:0">{escape(form)} · {escape(form_label)} · Filed {escape(filed)}</p>'
            f'<p style="color:#8E8E93;font-size:12px;line-height:1.5;margin:8px 0 0">{summary_html}</p></div>'
            f'<div style="min-width:105px;text-align:right">{url_html}</div></div></div>',
            unsafe_allow_html=True,
        )

    if compact:
        st.caption(
            "A registration filing is a research lead, not confirmation that an IPO will price or list. "
            "Offering dates, valuation and dilution are shown only when disclosed in source filings."
        )
    else:
        st.markdown("---")
        explainer_cols = st.columns(3)
        explainer_cols[0].markdown("**S-1 / S-1/A**\n\nU.S. issuer registration statements and amendments.")
        explainer_cols[1].markdown("**F-1 / F-1/A**\n\nRegistration statements and amendments for certain foreign issuers.")
        explainer_cols[2].markdown("**Not an IPO calendar**\n\nA filing is a signal to investigate, not confirmation of a listing date or investment opportunity.")
        st.markdown("[Search all SEC filings](https://www.sec.gov/edgar/search/)")
