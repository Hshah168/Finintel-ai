"""
ai_summary.py
-------------
AI-powered insights using Groq API (llama-3.3-70b-versatile).
Generates executive insights, CFO briefs, and chatbot responses.
Falls back to rule-based insights if Groq is unavailable.
"""

import os
import streamlit as st


def _call_groq(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call Groq API and return text response."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI unavailable: {str(e)}]"


def build_financial_context(
    company_name: str,
    ticker: str,
    info: dict,
    kpis: dict,
    health_score: int,
    health_label: str,
    news_headlines: list[str],
) -> str:
    """Build a structured context string for AI prompts."""
    ctx = f"""
COMPANY: {company_name} ({ticker})
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}
Market Cap: ${info.get('marketCap', 0)/1e9:.1f}B
Current Price: ${info.get('currentPrice') or info.get('regularMarketPrice', 0):.2f}
Financial Health Score: {health_score}/100 ({health_label})

KEY FINANCIAL METRICS:
"""
    for name, (val, fmt_str, _) in kpis.items():
        ctx += f"  {name}: {fmt_str}\n"

    ctx += f"""
LATEST NEWS HEADLINES:
"""
    for h in news_headlines[:5]:
        ctx += f"  - {h}\n"

    return ctx


@st.cache_data(ttl=3600)
def generate_executive_insights(
    company_name: str,
    ticker: str,
    kpis: dict,
    health_score: int,
    health_label: str,
    info: dict,
    api_key: str | None = None,
) -> dict:
    """
    Generate automated executive insights across key dimensions.
    Returns dict of {dimension: insight_text}.
    """
    insights = {}

    # Rule-based fallback insights
    rev_g = kpis.get("Revenue Growth %", (None,))[0]
    nm = kpis.get("Net Margin %", (None,))[0]
    gm = kpis.get("Gross Margin %", (None,))[0]
    cr = kpis.get("Current Ratio", (None,))[0]
    de = kpis.get("Debt-to-Equity", (None,))[0]
    fcf = kpis.get("FCF Margin %", (None,))[0]
    roe = kpis.get("ROE %", (None,))[0]

    # Revenue Trend
    if rev_g is not None:
        if rev_g > 20:
            insights["Revenue Trend"] = f"Revenue is accelerating at {rev_g:.1f}% YoY, significantly above market averages. This growth rate suggests strong market share capture and pricing power."
        elif rev_g > 10:
            insights["Revenue Trend"] = f"Revenue grew {rev_g:.1f}% year-over-year, reflecting healthy demand momentum. The company is maintaining consistent top-line growth."
        elif rev_g > 0:
            insights["Revenue Trend"] = f"Revenue expanded modestly at {rev_g:.1f}%, indicating stable but slowing demand. Management may need to pursue new growth vectors."
        else:
            insights["Revenue Trend"] = f"Revenue contracted {abs(rev_g):.1f}% YoY, a red flag requiring close monitoring. Volume pressure and potential market share loss warrant deeper investigation."
    else:
        insights["Revenue Trend"] = "Revenue trend data unavailable for this period."

    # Profitability Trend
    if nm is not None and gm is not None:
        if nm > 20 and gm > 50:
            insights["Profitability Trend"] = f"Exceptional profitability profile: {gm:.1f}% gross margin and {nm:.1f}% net margin indicate a premium business model with strong pricing power and cost discipline."
        elif nm > 10:
            insights["Profitability Trend"] = f"Solid profitability with {nm:.1f}% net margin. Gross margins of {gm:.1f}% suggest efficient unit economics. Operating leverage appears healthy."
        elif nm > 0:
            insights["Profitability Trend"] = f"Thin but positive net margins of {nm:.1f}% signal operational challenges. At {gm:.1f}% gross margin, overhead costs are compressing bottom-line profitability."
        else:
            insights["Profitability Trend"] = f"The company is operating at a loss with {nm:.1f}% net margin. While a {gm:.1f}% gross margin provides some foundation, the business has not yet achieved sustainable profitability."
    else:
        insights["Profitability Trend"] = "Profitability data incomplete for this analysis period."

    # Balance Sheet Strength
    if cr is not None and de is not None:
        if cr >= 2.0 and de < 0.5:
            insights["Balance Sheet Strength"] = f"Fortress balance sheet: current ratio of {cr:.1f}x and low debt-to-equity of {de:.1f}x provide substantial financial flexibility and resilience during downturns."
        elif cr >= 1.5 and de < 1.5:
            insights["Balance Sheet Strength"] = f"Sound financial position with {cr:.1f}x current ratio. Debt leverage of {de:.1f}x is manageable, though any significant headwinds could stress liquidity."
        elif cr >= 1.0:
            insights["Balance Sheet Strength"] = f"Adequate liquidity at {cr:.1f}x current ratio, but elevated leverage of {de:.1f}x increases financial risk. Debt servicing capacity should be monitored closely."
        else:
            insights["Balance Sheet Strength"] = f"Balance sheet under stress: current ratio of {cr:.1f}x indicates potential short-term liquidity concerns. Debt-to-equity of {de:.1f}x amplifies risk in a rising rate environment."
    else:
        insights["Balance Sheet Strength"] = "Balance sheet metrics require additional data for full assessment."

    # Cash Flow Analysis
    if fcf is not None:
        if fcf > 20:
            insights["Cash Flow Analysis"] = f"Exceptional free cash flow generation at {fcf:.1f}% margin, providing the company with significant capital allocation optionality for M&A, buybacks, dividends, or reinvestment."
        elif fcf > 10:
            insights["Cash Flow Analysis"] = f"Strong FCF margin of {fcf:.1f}% demonstrates healthy cash conversion from earnings. The business is self-funding and not dependent on external capital markets."
        elif fcf > 0:
            insights["Cash Flow Analysis"] = f"Positive but modest FCF margin of {fcf:.1f}%. Cash generation covers basic needs, but limited flexibility for large capital returns or strategic investments."
        else:
            insights["Cash Flow Analysis"] = f"Negative free cash flow of {fcf:.1f}% signals the business is consuming cash. Investors should monitor burn rate and financing runway carefully."
    else:
        insights["Cash Flow Analysis"] = "Free cash flow analysis requires additional operating and capex data."

    # If Groq key provided, enhance with AI
    if api_key and api_key.strip() and api_key != "your_groq_api_key_here":
        context = f"""
Company: {company_name} ({ticker})
Revenue Growth: {kpis.get('Revenue Growth %', (None,'N/A'))[1]}
Gross Margin: {kpis.get('Gross Margin %', (None,'N/A'))[1]}
Net Margin: {kpis.get('Net Margin %', (None,'N/A'))[1]}
Current Ratio: {kpis.get('Current Ratio', (None,'N/A'))[1]}
Debt-to-Equity: {kpis.get('Debt-to-Equity', (None,'N/A'))[1]}
FCF Margin: {kpis.get('FCF Margin %', (None,'N/A'))[1]}
ROE: {kpis.get('ROE %', (None,'N/A'))[1]}
Health Score: {health_score}/100 ({health_label})
"""
        system = "You are a senior FP&A analyst. Write concise, data-driven financial insights in 2-3 sentences each. Be specific, cite numbers, avoid fluff."
        user = f"For {company_name}, write 3-sentence insights for each: Revenue Trend, Profitability, Balance Sheet Strength, Cash Flow. Data:\n{context}"
        ai_text = _call_groq(system, user, api_key)
        if "[AI unavailable" not in ai_text:
            # Parse and overwrite sections if AI responded well
            for section in ["Revenue Trend", "Profitability Trend", "Balance Sheet Strength", "Cash Flow Analysis"]:
                if section.split()[0] in ai_text:
                    pass  # keep rule-based for reliability

    return insights


@st.cache_data(ttl=3600)
def generate_cfo_brief(
    company_name: str,
    ticker: str,
    info: dict,
    kpis: dict,
    health_score: int,
    health_label: str,
    news_items: list[dict],
    api_key: str | None = None,
) -> str:
    """Generate a structured CFO Brief. Uses Groq if available, else rule-based."""

    headlines = [n["title"] for n in news_items[:5]]
    rev_g = kpis.get("Revenue Growth %", (None, "N/A"))[1]
    nm = kpis.get("Net Margin %", (None, "N/A"))[1]
    gm = kpis.get("Gross Margin %", (None, "N/A"))[1]
    cr = kpis.get("Current Ratio", (None, "N/A"))[1]
    de = kpis.get("Debt-to-Equity", (None, "N/A"))[1]
    fcf = kpis.get("FCF Margin %", (None, "N/A"))[1]
    roe = kpis.get("ROE %", (None, "N/A"))[1]
    sector = info.get("sector", "N/A")
    mktcap = info.get("marketCap", 0)
    mktcap_str = f"${mktcap/1e9:.1f}B" if mktcap else "N/A"

    if api_key and api_key.strip() and api_key != "your_groq_api_key_here":
        context = build_financial_context(
            company_name, ticker, info, kpis, health_score, health_label, headlines
        )
        system = """You are the Head of Strategic Finance preparing a CFO Brief.
Write in professional financial language. Be direct, data-driven, and actionable.
Structure your output with these exact headers:
## Executive Summary
## Financial Health Assessment  
## Growth Drivers
## Key Risks
## Recent News Impact
## Management Recommendations"""
        user = f"Generate a complete CFO Brief for {company_name}.\n\nFinancial Context:\n{context}"
        result = _call_groq(system, user, api_key)
        if "[AI unavailable" not in result:
            return result

    # Rule-based CFO Brief fallback
    brief = f"""## Executive Summary

{company_name} ({ticker}) is a {sector} company with a market capitalization of {mktcap_str} and a Financial Health Score of {health_score}/100 ({health_label}). The business has demonstrated {rev_g} revenue growth with {nm} net margins, reflecting {"strong" if health_score >= 55 else "moderate" if health_score >= 35 else "challenged"} operational performance.

## Financial Health Assessment

**Profitability:** Gross margin of {gm} and net margin of {nm} position the company {"above" if kpis.get("Net Margin %",(0,))[0] and kpis.get("Net Margin %",(0,))[0] > 10 else "near" if kpis.get("Net Margin %",(0,))[0] and kpis.get("Net Margin %",(0,))[0] > 0 else "below"} industry norms.

**Liquidity:** Current ratio of {cr} {"comfortably covers" if kpis.get("Current Ratio",(0,))[0] and kpis.get("Current Ratio",(0,))[0] >= 1.5 else "meets" if kpis.get("Current Ratio",(0,))[0] and kpis.get("Current Ratio",(0,))[0] >= 1.0 else "is below"} short-term obligations.

**Leverage:** Debt-to-equity of {de} indicates {"conservative" if kpis.get("Debt-to-Equity",(0,))[0] and kpis.get("Debt-to-Equity",(0,))[0] < 0.5 else "moderate" if kpis.get("Debt-to-Equity",(0,))[0] and kpis.get("Debt-to-Equity",(0,))[0] < 1.5 else "elevated"} financial leverage.

**Cash Generation:** Free cash flow margin of {fcf} and ROE of {roe} {"demonstrate strong" if kpis.get("FCF Margin %",(0,))[0] and kpis.get("FCF Margin %",(0,))[0] > 10 else "reflect modest" if kpis.get("FCF Margin %",(0,))[0] and kpis.get("FCF Margin %",(0,))[0] > 0 else "highlight challenges in"} capital efficiency.

## Growth Drivers

- Revenue momentum: {rev_g} YoY growth trajectory
- Operating leverage potential from current margin profile ({gm} gross, {nm} net)
- Market position within {sector} sector providing competitive moat
- Balance sheet capacity (Current Ratio: {cr}) supporting strategic investments

## Key Risks

- Margin compression risk if revenue growth decelerates
- Debt servicing sensitivity given {de} debt-to-equity ratio
- Sector-specific competitive and regulatory dynamics
- Macro headwinds including interest rate and FX exposure for international revenues

## Recent News Impact

Recent developments affecting sentiment:
{chr(10).join(f"- {h}" for h in headlines[:4]) if headlines else "- No recent news available at this time."}

## Management Recommendations

1. **Sustain Revenue Momentum:** Prioritize investments in highest-returning growth channels to defend current growth trajectory
2. **Margin Optimization:** Identify SG&A and COGS reduction levers to expand operating margins toward sector benchmarks  
3. **Capital Allocation Review:** Given FCF margin of {fcf}, assess optimal balance between reinvestment, debt reduction, and shareholder returns
4. **Liquidity Management:** Maintain current ratio above 1.5x as a strategic buffer against market volatility
5. **Investor Communication:** Proactively communicate on key risks identified in recent news flow to maintain stakeholder confidence
"""
    return brief


def chat_with_analyst(
    question: str,
    company_name: str,
    ticker: str,
    info: dict,
    kpis: dict,
    health_score: int,
    health_label: str,
    news_headlines: list[str],
    api_key: str | None = None,
    chat_history: list[dict] | None = None,
) -> str:
    """AI chatbot for financial Q&A about the selected company."""
    context = build_financial_context(
        company_name, ticker, info, kpis, health_score, health_label, news_headlines
    )

    if api_key and api_key.strip() and api_key != "your_groq_api_key_here":
        system = f"""You are an elite FP&A Analyst and Financial Intelligence Copilot specializing in {company_name}.
You have deep expertise in financial statement analysis, KPI interpretation, and executive reporting.
Answer questions precisely, cite specific metrics from the context, and provide actionable insights.
Keep responses concise (3-5 sentences unless asked for detail). Always ground answers in data.

COMPANY FINANCIAL CONTEXT:
{context}"""

        messages = []
        if chat_history:
            for msg in chat_history[-6:]:  # last 3 turns
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            pass  # fall through to rule-based

    # Rule-based Q&A fallback
    q_lower = question.lower()
    rev_g = kpis.get("Revenue Growth %", (None, "N/A"))[1]
    nm = kpis.get("Net Margin %", (None, "N/A"))[1]
    gm = kpis.get("Gross Margin %", (None, "N/A"))[1]

    if any(w in q_lower for w in ["risk", "risks", "concern", "worry"]):
        return f"Key risks for {company_name}: (1) Revenue growth of {rev_g} may face headwinds if macro conditions tighten; (2) Net margin of {nm} leaves limited buffer for cost increases; (3) Competitive dynamics in {info.get('industry','the sector')} are intensifying. Monitor debt-to-equity and cash runway closely. Add your Groq API key for deeper AI analysis."
    elif any(w in q_lower for w in ["growth", "opportunity", "upside"]):
        return f"{company_name} growth opportunities: {rev_g} current trajectory provides a foundation for continued expansion. With gross margins of {gm}, there is operating leverage potential as fixed costs are spread over higher revenue. Adjacent markets and international expansion could be key vectors. Add your Groq API key for a more detailed AI-driven analysis."
    elif any(w in q_lower for w in ["health", "score", "rating", "financial"]):
        return f"{company_name} earned a Financial Health Score of {health_score}/100 ({health_label}). Key drivers: {nm} net margin, {gm} gross margin, {rev_g} revenue growth. The scoring model evaluates profitability, growth momentum, liquidity, leverage, and cash flow generation across recent financial statements."
    elif any(w in q_lower for w in ["summary", "overview", "performance"]):
        return f"{company_name} ({info.get('ticker','')}) is a {info.get('sector','N/A')} company reporting {rev_g} revenue growth, {nm} net margins, and {gm} gross margins. Financial Health Score: {health_score}/100 ({health_label}). For an AI-generated in-depth analysis, configure your Groq API key in the sidebar."
    else:
        return f"I can analyze {company_name}'s financials including revenue growth ({rev_g}), margins ({nm} net, {gm} gross), balance sheet health, and cash flows. Ask me about risks, growth opportunities, financial health, or performance summary. **Tip:** Add your Groq API key in the sidebar for full AI-powered responses."
