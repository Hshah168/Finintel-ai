"""
variance_analysis.py
--------------------
Automated variance analysis — the most common FP&A deliverable.

Compares two periods (current vs prior year, or actual vs budget)
and decomposes variance into:
  - Volume effect (how much of the change is driven by top-line)
  - Mix/margin effect (how much is driven by cost structure changes)
  - Absolute dollar change and percentage change

Returns plain-English narratives for each line item, the way
an FP&A analyst would write them in a board deck.
"""

import pandas as pd
import numpy as np


def _safe_float(val):
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _fmt_val(v):
    """Format a dollar value for narrative."""
    if v is None:
        return "N/A"
    av = abs(v)
    if av >= 1e9:
        return f"${av/1e9:.1f}B"
    if av >= 1e6:
        return f"${av/1e6:.1f}M"
    return f"${av/1e3:.0f}K"


def _direction(v, positive_is_good=True):
    """Return favourable/unfavourable label."""
    if v is None:
        return ""
    if v > 0:
        return "favorable" if positive_is_good else "unfavorable"
    return "unfavorable" if positive_is_good else "favorable"


def _pct_change(curr, prior):
    if curr is None or prior is None or prior == 0:
        return None
    return (curr - prior) / abs(prior) * 100


# Key line items and whether an increase is good (True) or bad (False)
INCOME_ITEMS = [
    ("Total Revenue",             ["Total Revenue", "Revenue", "Net Revenue"], True),
    ("Gross Profit",              ["Gross Profit"],                             True),
    ("Operating Income",          ["Operating Income", "EBIT"],                 True),
    ("Net Income",                ["Net Income", "Net Income Common Stockholders"], True),
    ("Cost Of Revenue",           ["Cost Of Revenue", "Cost of Goods Sold"],    False),
    ("Operating Expense",         ["Operating Expense", "Total Operating Expenses"], False),
    ("Research & Development",    ["Research And Development"],                  None),  # neutral
    ("Selling General Admin",     ["Selling General Administrative"],            False),
]

BALANCE_ITEMS = [
    ("Total Assets",              ["Total Assets"],                              True),
    ("Total Current Assets",      ["Total Current Assets"],                      True),
    ("Cash",                      ["Cash And Cash Equivalents", "Cash"],          True),
    ("Total Debt",                ["Total Debt", "Long Term Debt"],              False),
    ("Total Equity",              ["Total Stockholder Equity"],                   True),
    ("Current Liabilities",       ["Total Current Liabilities"],                 False),
]

CASHFLOW_ITEMS = [
    ("Operating Cash Flow",       ["Operating Cash Flow", "Total Cash From Operating Activities"], True),
    ("Capital Expenditure",       ["Capital Expenditure"],                        None),
    ("Free Cash Flow",            ["Free Cash Flow"],                             True),
]


def _get_row(df, keys):
    """Extract most recent and prior year values from a statement DataFrame."""
    if df is None or df.empty or len(df.columns) < 2:
        return None, None
    col_curr  = df.columns[0]
    col_prior = df.columns[1]
    for k in keys:
        if k in df.index:
            curr  = _safe_float(df.loc[k, col_curr])
            prior = _safe_float(df.loc[k, col_prior])
            return curr, prior
    return None, None


def _year_label(df, col_idx):
    """Get year label from DataFrame column header."""
    if df is None or df.empty or col_idx >= len(df.columns):
        return "Current" if col_idx == 0 else "Prior"
    col = df.columns[col_idx]
    return str(col.year) if hasattr(col, "year") else str(col)[:4]


def build_variance_table(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict:
    """
    Build full variance analysis across all three statements.

    Returns:
        {
          "income":   list of variance row dicts,
          "balance":  list of variance row dicts,
          "cashflow": list of variance row dicts,
          "curr_year": str,
          "prior_year": str,
          "narratives": list of plain-English insight strings,
        }
    """
    curr_year  = _year_label(income, 0) if income is not None and not income.empty else "Current"
    prior_year = _year_label(income, 1) if income is not None and not income.empty else "Prior"

    def build_rows(df, items):
        rows = []
        for display_name, keys, positive_good in items:
            curr, prior = _get_row(df, keys)
            if curr is None and prior is None:
                continue
            delta     = (curr - prior) if (curr is not None and prior is not None) else None
            delta_pct = _pct_change(curr, prior)
            rows.append({
                "line_item":     display_name,
                "current":       curr,
                "prior":         prior,
                "delta":         delta,
                "delta_pct":     delta_pct,
                "positive_good": positive_good,
                "favorable":     (delta is not None and
                                  ((delta > 0 and positive_good) or
                                   (delta < 0 and positive_good is False))),
            })
        return rows

    income_rows   = build_rows(income,   INCOME_ITEMS)
    balance_rows  = build_rows(balance,  BALANCE_ITEMS)
    cashflow_rows = build_rows(cashflow, CASHFLOW_ITEMS)

    # Build plain-English narratives for top drivers
    narratives = _build_narratives(income_rows, balance_rows, cashflow_rows,
                                   curr_year, prior_year)

    return {
        "income":     income_rows,
        "balance":    balance_rows,
        "cashflow":   cashflow_rows,
        "curr_year":  curr_year,
        "prior_year": prior_year,
        "narratives": narratives,
    }


def _build_narratives(income_rows, balance_rows, cashflow_rows,
                      curr_year, prior_year):
    """Generate plain-English variance narratives the way an FP&A analyst writes them."""
    narratives = []
    row_map = {r["line_item"]: r for r in income_rows + balance_rows + cashflow_rows}

    # Revenue narrative
    rev = row_map.get("Total Revenue")
    gp  = row_map.get("Gross Profit")
    op  = row_map.get("Operating Income")
    ni  = row_map.get("Net Income")
    ocf = row_map.get("Operating Cash Flow")
    debt = row_map.get("Total Debt")
    cash = row_map.get("Cash")

    if rev and rev["delta"] is not None:
        direction = "grew" if rev["delta"] > 0 else "declined"
        narratives.append(
            f"Revenue {direction} {_fmt_val(abs(rev['delta']))} "
            f"({rev['delta_pct']:+.1f}%) from {prior_year} to {curr_year}, "
            f"reflecting {'strong demand and market share gains' if rev['delta'] > 0 else 'headwinds in core markets'}."
        )

    # Margin narrative — revenue growth vs gross profit growth
    if rev and gp and rev["delta_pct"] and gp["delta_pct"]:
        if gp["delta_pct"] > rev["delta_pct"] + 2:
            narratives.append(
                f"Gross margin expanded as gross profit grew {gp['delta_pct']:+.1f}% versus revenue growth of "
                f"{rev['delta_pct']:+.1f}%, indicating improving pricing power and cost efficiency."
            )
        elif gp["delta_pct"] < rev["delta_pct"] - 2:
            narratives.append(
                f"Gross margin compressed — gross profit grew only {gp['delta_pct']:+.1f}% against revenue growth of "
                f"{rev['delta_pct']:+.1f}%, suggesting rising input costs or pricing pressure."
            )
        else:
            narratives.append(
                f"Gross margin held relatively stable with gross profit and revenue growing in line "
                f"({gp['delta_pct']:+.1f}% vs {rev['delta_pct']:+.1f}%)."
            )

    # Operating income narrative
    if op and op["delta"] is not None:
        direction = "increased" if op["delta"] > 0 else "decreased"
        driver = ""
        if rev and op and rev["delta_pct"] and op["delta_pct"]:
            if op["delta_pct"] > rev["delta_pct"]:
                driver = " Operating leverage is evident as operating income grew faster than revenue."
            elif op["delta_pct"] < rev["delta_pct"] - 5:
                driver = " Cost growth outpaced revenue, compressing operating leverage."
        narratives.append(
            f"Operating income {direction} {_fmt_val(abs(op['delta']))} "
            f"({op['delta_pct']:+.1f}%) year-over-year.{driver}"
        )

    # Net income narrative
    if ni and op and ni["delta"] is not None and op["delta"] is not None:
        if abs(ni["delta_pct"] or 0) > abs(op["delta_pct"] or 0) + 5:
            narratives.append(
                f"Net income moved {ni['delta_pct']:+.1f}% versus operating income at {op['delta_pct']:+.1f}%, "
                f"suggesting {'tax benefits or one-time gains' if ni['delta'] > op['delta'] else 'below-the-line charges or higher tax expense'} impacted the bottom line."
            )

    # Cash narrative
    if ocf and ocf["delta"] is not None:
        direction = "strengthened" if ocf["delta"] > 0 else "weakened"
        narratives.append(
            f"Operating cash flow {direction} by {_fmt_val(abs(ocf['delta']))} "
            f"({ocf['delta_pct']:+.1f}%), "
            f"{'demonstrating strong cash conversion from earnings' if ocf['delta'] > 0 else 'reflecting working capital pressure or higher cash costs'}."
        )

    # Debt narrative
    if debt and debt["delta"] is not None and abs(debt["delta"] or 0) > 1e8:
        direction = "increased" if debt["delta"] > 0 else "decreased"
        narratives.append(
            f"Total debt {direction} by {_fmt_val(abs(debt['delta']))} year-over-year, "
            f"{'adding leverage to the balance sheet' if debt['delta'] > 0 else 'reflecting debt paydown and improving balance sheet quality'}."
        )

    return narratives


def format_variance_df(rows: list[dict], curr_year: str, prior_year: str) -> pd.DataFrame:
    """Convert variance rows into a display DataFrame."""
    if not rows:
        return pd.DataFrame()

    def fmt(v):
        if v is None:
            return "—"
        av = abs(v)
        if av >= 1e9:
            return f"${v/1e9:,.2f}B"
        if av >= 1e6:
            return f"${v/1e6:,.1f}M"
        return f"${v/1e3:,.0f}K"

    def fmt_pct(v):
        return f"{v:+.1f}%" if v is not None else "—"

    display = []
    for r in rows:
        display.append({
            "Line Item":          r["line_item"],
            curr_year:            fmt(r["current"]),
            prior_year:           fmt(r["prior"]),
            "$ Change":           fmt(r["delta"]),
            "% Change":           fmt_pct(r["delta_pct"]),
            "_favorable":         r.get("favorable"),
            "_delta":             r.get("delta"),
        })
    return pd.DataFrame(display)
