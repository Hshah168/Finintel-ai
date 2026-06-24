# FinIntel AI — Financial Intelligence Platform

> Enterprise-grade financial intelligence for FP&A teams, analysts, and business leaders.

Built by **Hetal Shah** · [github.com/Hshah168](https://github.com/Hshah168)

---

## Features

| Module | Description |
|--------|-------------|
| 🔍 Smart Search | Type any company name — auto-resolves to ticker |
| 📈 Price Analytics | Candlestick charts with 20d MA, 5 time periods |
| 📋 Financial Statements | Income, Balance Sheet, Cash Flow with CSV export |
| 🎯 KPI Analysis | 11 KPIs including margins, liquidity, leverage, ROE |
| 🏥 Health Score | 0-100 score across 5 dimensions with radar chart |
| 💡 Executive Insights | Automated FP&A narrative on 4 key dimensions |
| 📄 CFO Brief | One-click structured executive brief generation |
| 📰 News Center | Real-time news with publisher and date |
| 🤖 AI Copilot | Financial Q&A chatbot (Groq-powered or rule-based) |
| 🏆 Peer Comparison | Side-by-side competitor benchmarking |

---

## Quickstart (Local)

```bash
# 1. Clone or download project
git clone https://github.com/Hshah168/finintel-ai
cd finintel-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## AI Configuration (Optional)

For AI-powered insights and chatbot responses, add a **Groq API key**:

1. Get a free key at [console.groq.com](https://console.groq.com)
2. Paste it into the sidebar "AI Settings" field at runtime

Or set it as an environment variable:
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

Without a key, the app functions fully using rule-based financial analysis.

---

## Streamlit Cloud Deployment

1. Push to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add secrets (optional):
   ```toml
   # .streamlit/secrets.toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Deploy — live in ~2 minutes

---

## Project Structure

```
finintel-ai/
├── app.py                 # Main Streamlit application
├── company_search.py      # Name-to-ticker resolution
├── financials.py          # Financial data retrieval
├── kpi_calculations.py    # KPI and health score engine
├── news.py                # News aggregation
├── ai_summary.py          # AI insights and CFO brief
├── utils.py               # Charts, formatting, UI components
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Tech Stack

- **Frontend:** Streamlit
- **Data:** Yahoo Finance (yfinance)
- **Charts:** Plotly
- **AI:** Groq API (llama-3.3-70b-versatile)
- **Analysis:** Pandas, NumPy

---

## Supported Companies

- **US:** All NYSE/NASDAQ listed companies
- **India:** NSE listed (e.g. TCS.NS, RELIANCE.NS, INFY)
- **Europe:** SAP, ASML, Nestlé, Volkswagen, etc.
- **Asia:** Sony, Toyota, Alibaba, TSMC, Samsung

---

## Disclaimer

Data sourced from Yahoo Finance. This platform is for **educational and analytical purposes only** and does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions.

---

*FinIntel AI · Built for portfolio demonstration · Not for commercial use*
