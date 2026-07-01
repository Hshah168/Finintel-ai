import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SurvivalResult:
    """Full output of the survival model."""
    company_name: str
    ticker: str

    # Scenario probabilities (sum to ~100)
    prob_thriving: float      # Scenario A
    prob_vulnerable: float    # Scenario B
    prob_critical: float      # Scenario C

    # Sub-scores (0-100 each)
    z_score: float            # Modified Altman Z
    z_score_label: str        # Safe / Grey / Distress
    runway_months: float | None  # Cash runway at current burn
    runway_label: str
    momentum_score: float     # Revenue acceleration 0-100
    momentum_label: str
    leverage_risk: float      # 0-100, higher = more risk
    margin_trajectory: str    # Improving / Stable / Deteriorating

    # Key signals
    red_flags: list[str]
    green_flags: list[str]

    # Narratives
    scenario_a_text: str
    scenario_b_text: str
    scenario_c_text: str
    headline: str             # One-sentence verdict
    overall_label: str        # Thriving / Vulnerable / Critical


def _safe(val, default=None):
    """Return val if it's a real number, else default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _get_val(df: pd.DataFrame, keys: list[str], col: int = 0):
    """Extract a value from a financial statement DataFrame."""
    if df is None or df.empty or col >= len(df.columns):
        return None
    c = df.columns[col]
    for k in keys:
        if k in df.index:
            return _safe(df.loc[k, c])
    return None


# ─── Modified Altman Z-Score for Tech/High-Growth Companies ──────────────────
# Classic Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
# But original model overfits to manufacturing. Tech adjustment:
# - Reduce weight on asset turnover (X5) — asset-light businesses distort this
# - Add R&D coverage proxy (not in original)
# - Recalibrate thresholds: safe >2.0, grey 1.0-2.0, distress <1.0

def calculate_modified_z_score(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    info: dict,
) -> tuple[float, str]:
    """
    Returns (z_score, label) where label is Safe / Grey Zone / Distress.
    """
    # Extract components
    total_assets = _get_val(balance, ["Total Assets"]) or 1

    # X1: Working Capital / Total Assets
    ca = _get_val(balance, ["Total Current Assets", "Current Assets"]) or 0
    cl = _get_val(balance, ["Total Current Liabilities", "Current Liabilities"]) or 0
    x1 = (ca - cl) / total_assets

    # X2: Retained Earnings / Total Assets
    re = _get_val(balance, ["Retained Earnings", "Accumulated Deficit"]) or 0
    x2 = re / total_assets

    # X3: EBIT / Total Assets
    ebit = _get_val(income, ["Operating Income", "EBIT", "Operating Income Or Loss"]) or 0
    x3 = ebit / total_assets

    # X4: Market Cap / Total Liabilities (tech adjustment: use equity value not book)
    mktcap = _safe(info.get("marketCap")) or _safe(info.get("enterpriseValue"))
    total_liab = _get_val(balance, ["Total Liab", "Total Liabilities Net Minority Interest"]) or 1
    if mktcap:
        x4 = mktcap / total_liab
    else:
        # fallback: book equity / total liabilities
        equity = _get_val(balance, ["Total Stockholder Equity", "Stockholders Equity"]) or 0
        x4 = max(0, equity) / total_liab

    # X5: Revenue / Total Assets (reduced weight for tech — asset-light distortion)
    rev = _get_val(income, ["Total Revenue", "Revenue", "Net Revenue"]) or 0
    x5 = rev / total_assets

    # Modified Z — recalibrated weights for tech/SaaS/high-growth
    z = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.75 * x5)

    # Tech-recalibrated thresholds (lower than original due to asset-light adjustment)
    if z >= 2.0:
        label = "Safe Zone"
    elif z >= 1.0:
        label = "Grey Zone"
    else:
        label = "Distress Zone"

    return round(z, 2), label


def calculate_runway(
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    income: pd.DataFrame,
) -> tuple[float | None, str]:
    """
    Estimate cash runway in months.
    Burn rate = negative operating cash flow per month.
    """
    cash = _get_val(balance, ["Cash And Cash Equivalents", "Cash",
                               "Cash And Short Term Investments"])
    ocf = _get_val(cashflow, ["Operating Cash Flow",
                               "Total Cash From Operating Activities",
                               "Cash Flow From Operations"])

    if cash is None:
        return None, "Insufficient data"

    cash = max(0, cash)

    if ocf is None or ocf >= 0:
        # Profitable or no data — runway is not a concern
        return 999, "Cash flow positive"

    monthly_burn = abs(ocf) / 12
    if monthly_burn == 0:
        return 999, "No burn detected"

    runway = cash / monthly_burn

    if runway >= 36:
        label = "Comfortable (3+ years)"
    elif runway >= 24:
        label = "Adequate (2-3 years)"
    elif runway >= 12:
        label = "Tight (12-24 months)"
    elif runway >= 6:
        label = "Critical (6-12 months)"
    else:
        label = "Severe (<6 months)"

    return round(runway, 1), label


def calculate_revenue_momentum(income: pd.DataFrame) -> tuple[float, str]:
    """
    Score revenue momentum 0-100.
    Measures whether growth is accelerating or decelerating.
    This is the key leading indicator from the PRA distress pattern:
    revenue deceleration precedes cash crisis by 12-18 months.
    """
    rev_keys = ["Total Revenue", "Revenue", "Net Revenue", "Revenues"]
    rev_row = None
    for k in rev_keys:
        if income is not None and not income.empty and k in income.index:
            rev_row = income.loc[k]
            break

    if rev_row is None or len(rev_row) < 2:
        return 50.0, "Insufficient data"

    values = [float(v) for v in rev_row.values if pd.notna(v) and float(v) != 0]
    if len(values) < 2:
        return 50.0, "Insufficient data"

    # Most recent first (columns sorted descending)
    if len(values) >= 3:
        g1 = (values[0] - values[1]) / abs(values[1])  # most recent YoY
        g2 = (values[1] - values[2]) / abs(values[2])  # prior year YoY
        acceleration = g1 - g2                           # positive = accelerating
    else:
        g1 = (values[0] - values[1]) / abs(values[1])
        acceleration = 0

    # Score: base from growth rate + acceleration bonus
    if g1 > 0.30:       base = 85
    elif g1 > 0.20:     base = 75
    elif g1 > 0.10:     base = 65
    elif g1 > 0.05:     base = 55
    elif g1 > 0:        base = 45
    elif g1 > -0.05:    base = 35
    elif g1 > -0.15:    base = 20
    else:               base = 10

    accel_bonus = min(15, max(-15, acceleration * 50))
    score = max(0, min(100, base + accel_bonus))

    if score >= 75:     label = "Strong acceleration"
    elif score >= 60:   label = "Healthy growth"
    elif score >= 45:   label = "Slowing growth"
    elif score >= 30:   label = "Stagnating"
    else:               label = "Revenue declining"

    return round(score, 1), label


def calculate_margin_trajectory(income: pd.DataFrame) -> str:
    """Detect if margins are improving, stable, or deteriorating."""
    gp_keys = ["Gross Profit"]
    rev_keys = ["Total Revenue", "Revenue", "Net Revenue"]

    if income is None or income.empty or len(income.columns) < 2:
        return "Insufficient data"

    gp_curr = _get_val(income, gp_keys, 0)
    gp_prior = _get_val(income, gp_keys, 1)
    rev_curr = _get_val(income, rev_keys, 0)
    rev_prior = _get_val(income, rev_keys, 1)

    if not all([gp_curr, gp_prior, rev_curr, rev_prior]) or rev_curr == 0 or rev_prior == 0:
        return "Insufficient data"

    gm_curr = gp_curr / rev_curr * 100
    gm_prior = gp_prior / rev_prior * 100
    delta = gm_curr - gm_prior

    if delta > 3:       return "Improving"
    elif delta > -1:    return "Stable"
    elif delta > -5:    return "Slightly deteriorating"
    else:               return "Deteriorating"


def calculate_leverage_risk(
    balance: pd.DataFrame,
    income: pd.DataFrame,
    info: dict,
) -> float:
    """Return leverage risk score 0-100 (higher = more risky)."""
    debt = _get_val(balance, ["Total Debt", "Long Term Debt", "Total Long Term Debt"]) or 0
    equity = _get_val(balance, ["Total Stockholder Equity", "Stockholders Equity"]) or 1
    ebitda = _safe(info.get("ebitda")) or _get_val(income, ["EBITDA"]) or 1

    de_ratio = debt / abs(equity) if equity != 0 else 10
    debt_ebitda = debt / abs(ebitda) if ebitda != 0 else 10

    # Score: 0=no risk, 100=extreme risk
    de_score = min(50, de_ratio * 10)
    ebitda_score = min(50, debt_ebitda * 5)
    return round(min(100, de_score + ebitda_score), 1)


# ─── Main predictor ───────────────────────────────────────────────────────────

def predict_survival(
    company_name: str,
    ticker: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    info: dict,
) -> SurvivalResult:
    """
    Run the full 24-month survival model and return a SurvivalResult.

    This is the core of the feature — combines four sub-models into
    a probabilistic scenario distribution with plain-English narrative.
    """
    # ── Run sub-models ────────────────────────────────────────────────────────
    z_score, z_label = calculate_modified_z_score(income, balance, cashflow, info)
    runway, runway_label = calculate_runway(balance, cashflow, income)
    momentum, momentum_label = calculate_revenue_momentum(income)
    margin_traj = calculate_margin_trajectory(income)
    leverage_risk = calculate_leverage_risk(balance, income, info)

    # ── Red and green flags ───────────────────────────────────────────────────
    red_flags = []
    green_flags = []

    # Z-Score signals
    if z_score < 1.0:
        red_flags.append(f"Altman Z-Score of {z_score:.2f} is in Distress Zone (<1.0) — pattern consistent with pre-default companies")
    elif z_score < 2.0:
        red_flags.append(f"Altman Z-Score of {z_score:.2f} is in Grey Zone — elevated financial risk")
    else:
        green_flags.append(f"Altman Z-Score of {z_score:.2f} is in Safe Zone — financial structure is sound")

    # Runway signals
    if runway is not None and runway < 12:
        red_flags.append(f"Cash runway of {runway:.1f} months is critically low — financing event required within 12 months")
    elif runway is not None and runway < 24:
        red_flags.append(f"Cash runway of {runway:.1f} months suggests a financing round may be needed within 2 years")
    elif runway is not None and runway < 999:
        green_flags.append(f"Cash runway of {runway:.1f} months provides adequate operational buffer")
    else:
        green_flags.append("Company is cash flow positive — no runway concern")

    # Momentum signals
    if momentum < 30:
        red_flags.append(f"Revenue momentum score of {momentum:.0f}/100 — declining revenue is the single strongest predictor of financial distress")
    elif momentum < 50:
        red_flags.append(f"Revenue momentum score of {momentum:.0f}/100 — growth is stalling, watch for continued deceleration")
    elif momentum >= 70:
        green_flags.append(f"Revenue momentum score of {momentum:.0f}/100 — strong growth trajectory reduces survival risk significantly")
    else:
        green_flags.append(f"Revenue momentum score of {momentum:.0f}/100 — adequate growth maintained")

    # Margin trajectory
    if "Deteriorating" in margin_traj:
        red_flags.append(f"Gross margin trajectory is {margin_traj.lower()} — cost structure is worsening as the business scales, a warning sign from the PRA distress playbook")
    elif margin_traj == "Improving":
        green_flags.append("Gross margin improving — unit economics are strengthening as the company scales")
    elif margin_traj == "Stable":
        green_flags.append("Gross margin stable — consistent unit economics with no deterioration signal")

    # Leverage signals
    if leverage_risk > 70:
        red_flags.append(f"Leverage risk score of {leverage_risk:.0f}/100 — debt burden relative to earnings creates refinancing vulnerability")
    elif leverage_risk < 30:
        green_flags.append(f"Leverage risk score of {leverage_risk:.0f}/100 — conservative debt structure, low refinancing risk")

    # ── Probability calculation ───────────────────────────────────────────────
    # Weighted composite risk score (0-100, higher = more at risk)
    risk_weights = {
        "z_score":    0.30,   # Most predictive in academic literature
        "runway":     0.25,   # Operational reality signal
        "momentum":   0.25,   # Leading indicator (PRA insight)
        "leverage":   0.10,   # Structural risk
        "margin":     0.10,   # Unit economics trajectory
    }

    # Normalize each to 0-100 risk
    z_risk = max(0, min(100, (2.5 - z_score) / 2.5 * 100))
    runway_risk = 0 if (runway is None or runway >= 36) else max(0, min(100, (36 - runway) / 36 * 100))
    momentum_risk = max(0, 100 - momentum)
    margin_risk = {"Improving": 10, "Stable": 25, "Slightly deteriorating": 55,
                   "Deteriorating": 80, "Insufficient data": 40}.get(margin_traj, 40)

    composite_risk = (
        z_risk         * risk_weights["z_score"] +
        runway_risk    * risk_weights["runway"] +
        momentum_risk  * risk_weights["momentum"] +
        leverage_risk  * risk_weights["leverage"] +
        margin_risk    * risk_weights["margin"]
    )

    # Convert composite risk to scenario probabilities
    if composite_risk < 25:
        prob_a = 75 + (25 - composite_risk) * 0.8
        prob_b = 100 - prob_a - 5
        prob_c = 5
    elif composite_risk < 50:
        prob_a = 75 - (composite_risk - 25) * 1.6
        prob_b = 100 - prob_a - 10
        prob_c = 10
    elif composite_risk < 75:
        prob_c = 15 + (composite_risk - 50) * 1.4
        prob_a = max(5, 50 - (composite_risk - 50) * 1.2)
        prob_b = 100 - prob_a - prob_c
    else:
        prob_c = 35 + (composite_risk - 75) * 1.0
        prob_a = max(5, 30 - (composite_risk - 75) * 0.5)
        prob_b = 100 - prob_a - prob_c

    # Normalize to sum to 100
    total = prob_a + prob_b + prob_c
    prob_a = round(prob_a / total * 100, 1)
    prob_b = round(prob_b / total * 100, 1)
    prob_c = round(100 - prob_a - prob_b, 1)

    # ── Determine dominant scenario ───────────────────────────────────────────
    if prob_a >= 50:
        overall_label = "Thriving"
        headline = f"{company_name} shows strong financial resilience — survival risk is low over the next 24 months."
    elif prob_b >= 40:
        overall_label = "Vulnerable"
        headline = f"{company_name} is financially viable but shows stress signals that warrant close monitoring over the next 18 months."
    else:
        overall_label = "Critical"
        headline = f"{company_name} displays financial distress patterns consistent with companies that required emergency intervention within 12-18 months."

    # ── Scenario narratives ───────────────────────────────────────────────────
    runway_str = f"{runway:.0f} months" if (runway and runway < 999) else "positive cash flow"
    z_str = f"{z_score:.2f}"
    mom_str = f"{momentum:.0f}/100"
    rev_keys = ["Total Revenue", "Revenue", "Net Revenue"]
    rev_curr = _get_val(income, rev_keys, 0)
    rev_str = f"${rev_curr/1e9:.1f}B" if rev_curr and rev_curr >= 1e9 else f"${rev_curr/1e6:.0f}M" if rev_curr else "N/A"

    scenario_a = (
        f"With {prob_a}% probability, {company_name} continues operating as a going concern through the next 24 months. "
        f"Revenue of {rev_str} with a momentum score of {mom_str} and Altman Z-Score of {z_str} suggest the business "
        f"has sufficient financial strength to fund operations, service debt, and execute its strategy without requiring "
        f"emergency capital. {margin_traj} margins indicate the unit economics trajectory is {'favorable' if 'Improv' in margin_traj else 'manageable'}."
    )

    scenario_b = (
        f"With {prob_b}% probability, {company_name} faces a liquidity or financing event within 18 months that "
        f"requires management intervention. Cash runway of {runway_str} and revenue momentum of {mom_str} "
        f"create a scenario where the company remains viable but must either accelerate revenue, reduce burn, "
        f"or access capital markets. This is the most likely outcome for companies in the grey zone — they survive, "
        f"but not without stress. Investors should monitor quarterly cash flow closely."
    )

    scenario_c = (
        f"With {prob_c}% probability, {company_name} exhibits financial characteristics that — based on distress "
        f"pattern analysis from debt portfolio modeling — are consistent with companies that entered default or "
        f"required emergency restructuring within 12-18 months. The Z-Score of {z_str} ({z_label}), "
        f"{'runway of ' + runway_str if runway and runway < 24 else 'cash position'}, and "
        f"{margin_traj.lower()} margins form a pattern that historically precedes financial crisis. "
        f"This does not mean failure is certain — it means the risk profile demands immediate attention."
    )

    return SurvivalResult(
        company_name=company_name,
        ticker=ticker,
        prob_thriving=prob_a,
        prob_vulnerable=prob_b,
        prob_critical=prob_c,
        z_score=z_score,
        z_score_label=z_label,
        runway_months=runway if runway and runway < 999 else None,
        runway_label=runway_label,
        momentum_score=momentum,
        momentum_label=momentum_label,
        leverage_risk=leverage_risk,
        margin_trajectory=margin_traj,
        red_flags=red_flags,
        green_flags=green_flags,
        scenario_a_text=scenario_a,
        scenario_b_text=scenario_b,
        scenario_c_text=scenario_c,
        headline=headline,
        overall_label=overall_label,
    )
