"""
kpi_calculations.py
-------------------
Derives financial KPIs from income statement, balance sheet,
and cash flow DataFrames. Returns clean dicts for display.
"""

import pandas as pd
import numpy as np


def _safe_get(df: pd.DataFrame, keys: list, col_idx: int = 0):
    """Try multiple row labels; return value from column col_idx."""
    if df is None or df.empty:
        return None
    col = df.columns[col_idx] if col_idx < len(df.columns) else None
    if col is None:
        return None
    for key in keys:
        if key in df.index:
            val = df.loc[key, col]
            if pd.notna(val):
                return float(val)
    return None


def calculate_kpis(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    info: dict,
) -> dict:
    """
    Calculate comprehensive KPIs.

    Returns a dict with KPI name -> (value, formatted_string, delta_vs_prior_year).
    """
    kpis = {}

    # ---- Revenue ----
    revenue_keys = ["Total Revenue", "Revenue", "Net Revenue", "Revenues"]
    rev_curr = _safe_get(income, revenue_keys, 0)
    rev_prior = _safe_get(income, revenue_keys, 1)

    if rev_curr and rev_prior and rev_prior != 0:
        rev_growth = (rev_curr - rev_prior) / abs(rev_prior) * 100
        kpis["Revenue Growth %"] = (rev_growth, f"{rev_growth:+.1f}%", None)
    else:
        kpis["Revenue Growth %"] = (None, "N/A", None)

    # ---- Gross Margin ----
    gross_profit_keys = ["Gross Profit"]
    gp = _safe_get(income, gross_profit_keys, 0)
    if gp and rev_curr and rev_curr != 0:
        gm = gp / rev_curr * 100
        gp_prior = _safe_get(income, gross_profit_keys, 1)
        rev_prior2 = rev_prior
        delta = None
        if gp_prior and rev_prior2 and rev_prior2 != 0:
            gm_prior = gp_prior / rev_prior2 * 100
            delta = gm - gm_prior
        kpis["Gross Margin %"] = (gm, f"{gm:.1f}%", delta)
    else:
        kpis["Gross Margin %"] = (None, "N/A", None)

    # ---- Operating Margin ----
    op_keys = ["Operating Income", "EBIT", "Operating Income Or Loss"]
    op = _safe_get(income, op_keys, 0)
    if op is not None and rev_curr and rev_curr != 0:
        om = op / rev_curr * 100
        op_prior = _safe_get(income, op_keys, 1)
        delta = None
        if op_prior and rev_prior and rev_prior != 0:
            delta = om - (op_prior / rev_prior * 100)
        kpis["Operating Margin %"] = (om, f"{om:.1f}%", delta)
    else:
        kpis["Operating Margin %"] = (None, "N/A", None)

    # ---- Net Margin ----
    net_keys = [
        "Net Income", "Net Income Common Stockholders",
        "Net Income From Continuing Operations"
    ]
    ni = _safe_get(income, net_keys, 0)
    if ni is not None and rev_curr and rev_curr != 0:
        nm = ni / rev_curr * 100
        ni_prior = _safe_get(income, net_keys, 1)
        delta = None
        if ni_prior and rev_prior and rev_prior != 0:
            delta = nm - (ni_prior / rev_prior * 100)
        kpis["Net Margin %"] = (nm, f"{nm:.1f}%", delta)
    else:
        kpis["Net Margin %"] = (None, "N/A", None)

    # ---- EBITDA Margin ----
    ebitda = info.get("ebitda")
    if ebitda and rev_curr and rev_curr != 0:
        em = ebitda / rev_curr * 100
        kpis["EBITDA Margin %"] = (em, f"{em:.1f}%", None)
    else:
        kpis["EBITDA Margin %"] = (None, "N/A", None)

    # ---- Liquidity ----
    ca_keys = ["Current Assets", "Total Current Assets"]
    cl_keys = ["Current Liabilities", "Total Current Liabilities"]
    ca = _safe_get(balance, ca_keys, 0)
    cl = _safe_get(balance, cl_keys, 0)
    inv_keys = ["Inventory"]
    inv = _safe_get(balance, inv_keys, 0) or 0

    if ca and cl and cl != 0:
        cr = ca / cl
        kpis["Current Ratio"] = (cr, f"{cr:.2f}x", None)
        qr = (ca - inv) / cl
        kpis["Quick Ratio"] = (qr, f"{qr:.2f}x", None)
    else:
        kpis["Current Ratio"] = (None, "N/A", None)
        kpis["Quick Ratio"] = (None, "N/A", None)

    # ---- Leverage ----
    debt_keys = ["Total Debt", "Long Term Debt", "Total Long Term Debt"]
    equity_keys = ["Total Stockholder Equity", "Stockholders Equity",
                   "Total Equity Gross Minority Interest"]
    total_debt = _safe_get(balance, debt_keys, 0)
    equity = _safe_get(balance, equity_keys, 0)

    if total_debt is not None and equity and equity != 0:
        de = total_debt / equity
        kpis["Debt-to-Equity"] = (de, f"{de:.2f}x", None)
    else:
        de_info = info.get("debtToEquity")
        if de_info:
            de = de_info / 100
            kpis["Debt-to-Equity"] = (de, f"{de:.2f}x", None)
        else:
            kpis["Debt-to-Equity"] = (None, "N/A", None)

    # ---- ROA ----
    ta_keys = ["Total Assets"]
    ta = _safe_get(balance, ta_keys, 0)
    if ni is not None and ta and ta != 0:
        roa = ni / ta * 100
        kpis["ROA %"] = (roa, f"{roa:.1f}%", None)
    else:
        roa_info = info.get("returnOnAssets")
        if roa_info:
            kpis["ROA %"] = (roa_info * 100, f"{roa_info*100:.1f}%", None)
        else:
            kpis["ROA %"] = (None, "N/A", None)

    # ---- ROE ----
    if ni is not None and equity and equity != 0:
        roe = ni / equity * 100
        kpis["ROE %"] = (roe, f"{roe:.1f}%", None)
    else:
        roe_info = info.get("returnOnEquity")
        if roe_info:
            kpis["ROE %"] = (roe_info * 100, f"{roe_info*100:.1f}%", None)
        else:
            kpis["ROE %"] = (None, "N/A", None)

    # ---- Free Cash Flow Margin ----
    ocf_keys = ["Operating Cash Flow", "Total Cash From Operating Activities"]
    capex_keys = ["Capital Expenditure", "Capital Expenditures", "Purchase Of Plant And Equipment"]
    ocf = _safe_get(cashflow, ocf_keys, 0)
    capex = _safe_get(cashflow, capex_keys, 0)
    if ocf and capex is not None and rev_curr and rev_curr != 0:
        fcf = ocf + capex  # capex is usually negative
        fcf_margin = fcf / rev_curr * 100
        kpis["FCF Margin %"] = (fcf_margin, f"{fcf_margin:.1f}%", None)
    else:
        kpis["FCF Margin %"] = (None, "N/A", None)

    return kpis


def calculate_health_score(kpis: dict, info: dict) -> tuple[int, str, dict]:
    """
    Build a 0-100 Financial Health Score across 5 dimensions.

    Returns (score, label, breakdown_dict).
    """
    scores = {
        "Profitability": 0,
        "Growth": 0,
        "Liquidity": 0,
        "Leverage": 0,
        "Cash Flow": 0,
    }
    max_scores = {k: 20 for k in scores}  # each dimension max 20

    # Profitability (max 20)
    nm_val = kpis.get("Net Margin %", (None,))[0]
    gm_val = kpis.get("Gross Margin %", (None,))[0]
    if nm_val is not None:
        if nm_val > 25: scores["Profitability"] += 10
        elif nm_val > 15: scores["Profitability"] += 8
        elif nm_val > 5: scores["Profitability"] += 5
        elif nm_val > 0: scores["Profitability"] += 2
    if gm_val is not None:
        if gm_val > 60: scores["Profitability"] += 10
        elif gm_val > 40: scores["Profitability"] += 7
        elif gm_val > 20: scores["Profitability"] += 4
        elif gm_val > 0: scores["Profitability"] += 1

    # Growth (max 20)
    rev_g = kpis.get("Revenue Growth %", (None,))[0]
    if rev_g is not None:
        if rev_g > 30: scores["Growth"] += 20
        elif rev_g > 20: scores["Growth"] += 16
        elif rev_g > 10: scores["Growth"] += 12
        elif rev_g > 5: scores["Growth"] += 8
        elif rev_g > 0: scores["Growth"] += 4

    # Liquidity (max 20)
    cr_val = kpis.get("Current Ratio", (None,))[0]
    qr_val = kpis.get("Quick Ratio", (None,))[0]
    if cr_val is not None:
        if cr_val >= 2.0: scores["Liquidity"] += 10
        elif cr_val >= 1.5: scores["Liquidity"] += 8
        elif cr_val >= 1.0: scores["Liquidity"] += 5
        else: scores["Liquidity"] += 1
    if qr_val is not None:
        if qr_val >= 1.5: scores["Liquidity"] += 10
        elif qr_val >= 1.0: scores["Liquidity"] += 7
        elif qr_val >= 0.7: scores["Liquidity"] += 4
        else: scores["Liquidity"] += 1

    # Leverage (max 20)
    de_val = kpis.get("Debt-to-Equity", (None,))[0]
    if de_val is not None:
        if de_val < 0.3: scores["Leverage"] += 20
        elif de_val < 0.7: scores["Leverage"] += 15
        elif de_val < 1.5: scores["Leverage"] += 10
        elif de_val < 3.0: scores["Leverage"] += 5
        else: scores["Leverage"] += 0
    else:
        scores["Leverage"] += 10  # no debt info = neutral

    # Cash Flow (max 20)
    fcf_val = kpis.get("FCF Margin %", (None,))[0]
    roe_val = kpis.get("ROE %", (None,))[0]
    if fcf_val is not None:
        if fcf_val > 20: scores["Cash Flow"] += 12
        elif fcf_val > 10: scores["Cash Flow"] += 9
        elif fcf_val > 0: scores["Cash Flow"] += 5
    if roe_val is not None:
        if roe_val > 30: scores["Cash Flow"] += 8
        elif roe_val > 15: scores["Cash Flow"] += 6
        elif roe_val > 0: scores["Cash Flow"] += 3

    total = sum(scores.values())

    if total >= 75:
        label = "Excellent"
    elif total >= 55:
        label = "Strong"
    elif total >= 35:
        label = "Average"
    else:
        label = "Weak"

    return total, label, scores
