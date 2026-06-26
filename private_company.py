"""
private_company.py
------------------
Parses financial data from uploaded Excel/CSV/PDF files for private companies.
Extracts income statement, balance sheet, and cash flow data into
standardized DataFrames compatible with the existing KPI and health score engine.

Supported formats:
  - Excel (.xlsx, .xls) — single or multi-sheet
  - CSV (.csv)          — single statement per file
  - PDF (.pdf)          — text-based (not scanned) annual reports
"""

import io
import re
import importlib
import pandas as pd
import numpy as np
import streamlit as st


# ─── Row label normalizer ──────────────────────────────────────────────────────
# Maps common variations of financial statement line items to standard keys
# that match what kpi_calculations.py expects.

LABEL_MAP = {
    # Revenue
    "revenue": "Total Revenue",
    "total revenue": "Total Revenue",
    "net revenue": "Total Revenue",
    "net sales": "Total Revenue",
    "total sales": "Total Revenue",
    "sales": "Total Revenue",
    "turnover": "Total Revenue",
    "total turnover": "Total Revenue",
    "income from operations": "Total Revenue",

    # Gross Profit
    "gross profit": "Gross Profit",
    "gross income": "Gross Profit",
    "gross margin": "Gross Profit",

    # Operating Income
    "operating income": "Operating Income",
    "operating profit": "Operating Income",
    "ebit": "Operating Income",
    "profit from operations": "Operating Income",
    "income from operations (ebit)": "Operating Income",

    # Net Income
    "net income": "Net Income",
    "net profit": "Net Income",
    "net earnings": "Net Income",
    "profit after tax": "Net Income",
    "pat": "Net Income",
    "net income after tax": "Net Income",
    "profit for the year": "Net Income",
    "net profit after tax": "Net Income",

    # EBITDA
    "ebitda": "EBITDA",

    # Cost of Revenue
    "cost of revenue": "Cost Of Revenue",
    "cost of goods sold": "Cost Of Revenue",
    "cogs": "Cost Of Revenue",
    "cost of sales": "Cost Of Revenue",

    # Operating Expenses
    "operating expenses": "Operating Expense",
    "total operating expenses": "Operating Expense",
    "opex": "Operating Expense",

    # Balance Sheet — Assets
    "total assets": "Total Assets",
    "current assets": "Total Current Assets",
    "total current assets": "Total Current Assets",
    "cash and cash equivalents": "Cash And Cash Equivalents",
    "cash": "Cash And Cash Equivalents",
    "inventory": "Inventory",
    "inventories": "Inventory",
    "accounts receivable": "Net Receivables",
    "trade receivables": "Net Receivables",

    # Balance Sheet — Liabilities
    "total liabilities": "Total Liab",
    "current liabilities": "Total Current Liabilities",
    "total current liabilities": "Total Current Liabilities",
    "accounts payable": "Accounts Payable",
    "trade payables": "Accounts Payable",
    "total debt": "Total Debt",
    "long term debt": "Long Term Debt",
    "long-term debt": "Long Term Debt",
    "short term debt": "Short Long Term Debt",
    "short-term debt": "Short Long Term Debt",

    # Balance Sheet — Equity
    "total equity": "Total Stockholder Equity",
    "shareholders equity": "Total Stockholder Equity",
    "stockholders equity": "Total Stockholder Equity",
    "total shareholders equity": "Total Stockholder Equity",
    "owners equity": "Total Stockholder Equity",
    "net worth": "Total Stockholder Equity",

    # Cash Flow
    "operating cash flow": "Operating Cash Flow",
    "cash from operations": "Operating Cash Flow",
    "net cash from operating activities": "Operating Cash Flow",
    "cash flow from operations": "Operating Cash Flow",
    "capital expenditure": "Capital Expenditure",
    "capital expenditures": "Capital Expenditure",
    "capex": "Capital Expenditure",
    "purchase of property": "Capital Expenditure",
    "free cash flow": "Free Cash Flow",
}


def _normalize_label(raw: str) -> str:
    """Clean and normalize a row label string."""
    if not raw:
        return ""
    cleaned = str(raw).strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)   # remove punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return LABEL_MAP.get(cleaned, raw.strip())


def _clean_value(val) -> float | None:
    """Convert a cell value to float, handling strings like '$1,234,567' or '(500)'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s in ("-", "--", "n/a", "na", "nil", ""):
        return None
    # Handle parentheses for negatives: (1,234) -> -1234
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # Remove currency symbols and commas
    s = re.sub(r"[$€£₹,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _detect_scale(df_raw: pd.DataFrame) -> float:
    """
    Auto-detect if values are in thousands, millions, or billions
    by looking at the magnitude of revenue-like rows.
    Returns multiplier to get to raw units.
    """
    rev_keywords = ["revenue", "sales", "turnover"]
    for idx in df_raw.index:
        label = str(idx).lower()
        if any(k in label for k in rev_keywords):
            for col in df_raw.columns:
                val = _clean_value(df_raw.loc[idx, col])
                if val and abs(val) > 0:
                    if abs(val) < 1_000:           # probably in billions already
                        return 1e9
                    elif abs(val) < 1_000_000:     # in millions
                        return 1e6
                    elif abs(val) < 1_000_000_000: # in thousands
                        return 1e3
                    else:                           # already in raw units
                        return 1.0
    return 1.0  # default: assume raw units


# ─── Excel / CSV Parser ────────────────────────────────────────────────────────

def parse_excel_csv(file_bytes: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """
    Parse an Excel or CSV file into up to 3 financial statement DataFrames.

    Returns dict with keys: 'income', 'balance', 'cashflow'
    Each value is a DataFrame with row labels as index, year columns as headers.
    Missing statements return empty DataFrames.
    """
    statements = {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}

    try:
        if filename.endswith(".csv"):
            raw = pd.read_csv(io.BytesIO(file_bytes), index_col=0, header=0)
            sheets = {"Sheet1": raw}
        else:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            sheets = {}
            for sheet_name in xl.sheet_names:
                try:
                    df = xl.parse(sheet_name, index_col=0, header=0)
                    if not df.empty:
                        sheets[sheet_name] = df
                except Exception:
                    continue

        for sheet_name, raw_df in sheets.items():
            sheet_lower = sheet_name.lower()
            stmt_type = None

            # Identify statement type from sheet name
            if any(k in sheet_lower for k in ["income", "p&l", "profit", "pnl", "p & l"]):
                stmt_type = "income"
            elif any(k in sheet_lower for k in ["balance", "bs", "position", "assets"]):
                stmt_type = "balance"
            elif any(k in sheet_lower for k in ["cash", "cf", "flow"]):
                stmt_type = "cashflow"

            # Parse the sheet
            parsed = _parse_statement_df(raw_df)

            if stmt_type and not parsed.empty:
                statements[stmt_type] = parsed
            elif parsed is not None and not parsed.empty:
                # Auto-detect from row labels if sheet name is ambiguous
                labels_lower = [str(l).lower() for l in parsed.index]
                if any("revenue" in l or "sales" in l for l in labels_lower):
                    if statements["income"].empty:
                        statements["income"] = parsed
                if any("total assets" in l or "current assets" in l for l in labels_lower):
                    if statements["balance"].empty:
                        statements["balance"] = parsed
                if any("operating" in l and "cash" in l for l in labels_lower):
                    if statements["cashflow"].empty:
                        statements["cashflow"] = parsed

    except Exception as e:
        st.warning(f"Could not fully parse file: {str(e)}")

    return statements


def _parse_statement_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw DataFrame from Excel/CSV into a standardized financial statement.
    Normalizes row labels, cleans values, handles scale detection.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Drop fully empty rows/cols
    raw = raw.dropna(how="all").dropna(axis=1, how="all")

    # Detect scale
    scale = _detect_scale(raw)

    # Build clean DataFrame
    clean_data = {}
    for col in raw.columns:
        col_label = str(col).strip()
        # Try to parse column header as a year
        year_match = re.search(r"(20\d{2}|19\d{2})", col_label)
        col_key = year_match.group(1) if year_match else col_label
        clean_data[col_key] = {}

    result_rows = {}
    for idx in raw.index:
        normalized = _normalize_label(str(idx))
        if not normalized:
            continue
        row_vals = {}
        for col in raw.columns:
            col_label = str(col).strip()
            year_match = re.search(r"(20\d{2}|19\d{2})", col_label)
            col_key = year_match.group(1) if year_match else col_label
            val = _clean_value(raw.loc[idx, col])
            if val is not None:
                # Apply scale if value looks like it needs it
                if scale > 1 and abs(val) < 1e9:
                    val = val * scale
            row_vals[col_key] = val
        if any(v is not None for v in row_vals.values()):
            result_rows[normalized] = row_vals

    if not result_rows:
        return pd.DataFrame()

    df = pd.DataFrame(result_rows).T
    # Sort columns by year descending (most recent first)
    try:
        df = df.reindex(sorted(df.columns, reverse=True), axis=1)
    except Exception:
        pass

    return df


# ─── PDF Parser ───────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """
    Extract financial tables from a PDF annual report.
    Uses pdfplumber for table extraction.
    Returns same dict format as parse_excel_csv.
    """
    statements = {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}

    try:
        pdfplumber = importlib.import_module("pdfplumber")

        all_tables = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 3:
                        all_tables.append(table)

        for table in all_tables:
            df = _table_to_df(table)
            if df is None or df.empty:
                continue

            labels_lower = [str(l).lower() for l in df.index]

            # Classify table by its row labels
            has_revenue = any("revenue" in l or "sales" in l or "turnover" in l for l in labels_lower)
            has_assets = any("total assets" in l or "current assets" in l for l in labels_lower)
            has_cashflow = any("operating" in l and ("cash" in l or "activit" in l) for l in labels_lower)

            if has_revenue and statements["income"].empty:
                statements["income"] = df
            elif has_assets and statements["balance"].empty:
                statements["balance"] = df
            elif has_cashflow and statements["cashflow"].empty:
                statements["cashflow"] = df

    except ImportError:
        st.error("pdfplumber not installed. Run: pip install pdfplumber")
    except Exception as e:
        st.warning(f"PDF parsing error: {str(e)}")

    return statements


def _table_to_df(table: list) -> pd.DataFrame | None:
    """Convert a raw pdfplumber table (list of lists) to a clean DataFrame."""
    if not table or len(table) < 2:
        return None

    try:
        # First row as header
        headers = [str(h).strip() if h else "" for h in table[0]]

        rows = {}
        for row in table[1:]:
            if not row or not row[0]:
                continue
            label = _normalize_label(str(row[0]))
            if not label:
                continue
            row_data = {}
            for i, val in enumerate(row[1:], 1):
                if i < len(headers):
                    col_key = headers[i]
                    year_match = re.search(r"(20\d{2}|19\d{2})", col_key)
                    col_key = year_match.group(1) if year_match else col_key
                    cleaned = _clean_value(val)
                    if cleaned is not None:
                        row_data[col_key] = cleaned
            if row_data:
                rows[label] = row_data

        if not rows:
            return None

        df = pd.DataFrame(rows).T
        try:
            df = df.reindex(sorted(df.columns, reverse=True), axis=1)
        except Exception:
            pass
        return df

    except Exception:
        return None


# ─── Main entry point ─────────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Master parser — routes to Excel/CSV or PDF parser based on file type.

    Returns:
        (statements_dict, detected_company_name)
        statements_dict keys: 'income', 'balance', 'cashflow'
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        statements = parse_pdf(file_bytes)
    elif filename.endswith((".xlsx", ".xls", ".csv")):
        statements = parse_excel_csv(file_bytes, filename)
    else:
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}, ""

    # Try to extract company name from filename
    company_name = uploaded_file.name
    company_name = re.sub(r"\.(xlsx|xls|csv|pdf)$", "", company_name, flags=re.IGNORECASE)
    company_name = re.sub(r"[_\-]", " ", company_name).strip().title()

    return statements, company_name


def get_available_statements(statements: dict) -> list[str]:
    """Return list of which statements were successfully parsed."""
    available = []
    if not statements["income"].empty:
        available.append("Income Statement")
    if not statements["balance"].empty:
        available.append("Balance Sheet")
    if not statements["cashflow"].empty:
        available.append("Cash Flow")
    return available
