import yfinance as yf
import requests
import streamlit as st

# --- Companies that are PRIVATE (not publicly traded) ---
PRIVATE_COMPANIES = {
    "spacex", "space x", "openai", "open ai", "stripe", "klarna",
    "bytedance", "tiktok", "chime", "revolut", "canva", "instacart",
    "epic games", "cargill", "koch industries", "ikea", "lego",
    "bloomberg", "mckinsey", "deloitte", "pwc", "kpmg", "ey",
    "ernst young", "bain", "bcg", "boston consulting",
}

# --- Curated alias map for fast, reliable lookups ---
COMPANY_ALIASES = {
    # US Tech
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
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "twitter": "X",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    "datadog": "DDOG",
    "servicenow": "NOW",
    "crowdstrike": "CRWD",
    "palo alto": "PANW",
    "palo alto networks": "PANW",
    "fortinet": "FTNT",
    "mongodb": "MDB",
    "cloudflare": "NET",
    "twilio": "TWLO",
    "okta": "OKTA",
    "hubspot": "HUBS",
    "zendesk": "ZEN",
    "dropbox": "DBX",
    "box": "BOX",
    "squarespace": "SQSP",
    "wix": "WIX",
    "godaddy": "GDDY",
    "match group": "MTCH",
    "roblox": "RBLX",
    "unity": "U",
    "electronic arts": "EA",
    "ea": "EA",
    "take two": "TTWO",
    "take-two": "TTWO",
    "activision": "ATVI",
    "ubisoft": "UBSFY",
    "dell": "DELL",
    "hp": "HPQ",
    "hewlett packard": "HPQ",
    "hpe": "HPE",
    "motorola": "MSI",
    "corning": "GLW",
    "applied materials": "AMAT",
    "lam research": "LRCX",
    "kla": "KLAC",
    "analog devices": "ADI",
    "texas instruments": "TXN",
    "micron": "MU",
    "western digital": "WDC",
    "seagate": "STX",
    "netapp": "NTAP",
    "veeva": "VEEV",
    "splunk": "SPLK",
    "elastic": "ESTC",
    "confluent": "CFLT",
    "hashicorp": "HCP",
    "gitlab": "GTLB",
    # US Auto
    "ford": "F",
    "ford motor": "F",
    "general motors": "GM",
    "gm": "GM",
    "stellantis": "STLA",
    "chrysler": "STLA",
    "jeep": "STLA",
    "rivian": "RIVN",
    "lucid": "LCID",
    "lucid motors": "LCID",
    "fisker": "FSR",
    "workday": "WDAY",
    "zoom": "ZM",
    "slack": "CRM",
    "shopify": "SHOP",
    "stripe": "STRP",
    # US Finance
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
    # US Healthcare / Pharma
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "abbvie": "ABBV",
    "merck": "MRK",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "unitedhealth": "UNH",
    "cvs": "CVS",
    # US Consumer / Retail
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "home depot": "HD",
    "nike": "NKE",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "starbucks": "SBUX",
    "coca cola": "KO",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "procter gamble": "PG",
    "procter & gamble": "PG",
    # US Energy / Industrial
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "boeing": "BA",
    "lockheed martin": "LMT",
    "caterpillar": "CAT",
    "3m": "MMM",
    # International — Japan (ADR or TSE)
    "toyota": "TM",
    "toyota motor": "TM",
    "nissan": "NSANY",           # Nissan Motor ADR
    "nissan motor": "NSANY",
    "honda": "HMC",              # Honda Motor ADR
    "honda motor": "HMC",
    "sony": "SONY",
    "sony group": "SONY",
    "panasonic": "PCRFY",
    "sharp": "SHCAY",
    "hitachi": "HTHIY",
    "mitsubishi": "MSBHF",
    "softbank": "SFTBY",
    "nintendo": "NTDOY",
    "keyence": "KYCCF",
    "fanuc": "FANUY",
    "subaru": "FUJHY",
    "mazda": "MZDAY",
    "suzuki": "SZKMY",
    "isuzu": "ISUZY",
    "denso": "DNZOY",
    "nidec": "NJDCY",
    "fujifilm": "FUJIY",
    "canon": "CAJ",
    "nikon": "NINOY",
    "olympus": "OCPNY",
    "ricoh": "RICOY",
    "kddi": "KDDIY",
    "ntt": "NTTYY",
    "docomo": "NTTYY",
    "rakuten": "RKUNY",
    "line": "LYV",
    # International — South Korea
    "samsung": "SSNLF",
    "samsung electronics": "SSNLF",
    "lg": "LGCLF",
    "lg electronics": "LGCLF",
    "hyundai": "HYMTF",
    "hyundai motor": "HYMTF",
    "kia": "KIMTF",
    "sk hynix": "HXSCL",
    "posco": "PKX",
    "sk telecom": "SKM",
    # International — China
    "alibaba": "BABA",
    "tencent": "TCEHY",
    "baidu": "BIDU",
    "jd": "JD",
    "jd.com": "JD",
    "pinduoduo": "PDD",
    "meituan": "MPNGF",
    "xiaomi": "XIACF",
    "lenovo": "LNVGY",
    "didi": "DIDIY",
    "netease": "NTES",
    "nio": "NIO",
    "xpeng": "XPEV",
    "li auto": "LI",
    "byd": "BYDDY",
    # International — Europe
    "sap": "SAP",
    "asml": "ASML",
    "lvmh": "LVMUY",
    "volkswagen": "VWAGY",
    "vw": "VWAGY",
    "bmw": "BMWYY",
    "bmw group": "BMWYY",
    "mercedes": "MBGYY",
    "mercedes-benz": "MBGYY",
    "mercedes benz": "MBGYY",
    "porsche": "POAHY",
    "stellantis": "STLA",
    "renault": "RNLSY",
    "peugeot": "STLA",
    "siemens": "SIEGY",
    "bosch": "BSWQY",
    "basf": "BASFY",
    "bayer": "BAYRY",
    "novartis": "NVS",
    "roche": "RHHBY",
    "astrazeneca": "AZN",
    "glaxosmithkline": "GSK",
    "gsk": "GSK",
    "sanofi": "SNY",
    "shell": "SHEL",
    "bp": "BP",
    "total": "TTE",
    "totalenergies": "TTE",
    "eni": "E",
    "hsbc": "HSBC",
    "barclays": "BCS",
    "lloyds": "LYG",
    "ubs": "UBS",
    "credit suisse": "CS",
    "deutsche bank": "DB",
    "allianz": "ALIZY",
    "axa": "AXAHY",
    "unilever": "UL",
    "nestle": "NSRGY",
    "nestle sa": "NSRGY",
    "diageo": "DEO",
    "heineken": "HEINY",
    "ab inbev": "BUD",
    "anheuser busch": "BUD",
    "philips": "PHG",
    "airbus": "EADSY",
    "rolls royce": "RYCEY",
    "rio tinto": "RIO",
    "bhp": "BHP",
    "glencore": "GLNCY",
    "arm": "ARM",
    "arm holdings": "ARM",
    "spotify": "SPOT",
    "adyen": "ADYEY",
    "prosus": "PROSY",
    # International — India
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "infosys": "INFY",
    "wipro": "WIT",
    "hcl": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici": "IBN",
    "icici bank": "IBN",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "axis bank": "AXISBANK.NS",
    "bajaj": "BAJFINANCE.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "tata motors": "TTM",
    "mahindra": "M&M.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "itc": "ITC.NS",
    "sun pharma": "SUNPHARMA.NS",
    "dr reddy": "RDY",
    "cipla": "CIPLA.NS",
    "adani": "ADANIENT.NS",
    "adani enterprises": "ADANIENT.NS",
    "ola": "OLACABS.NS",
    "zomato": "ZOMATO.NS",
    "paytm": "PAYTM.NS",
    "nykaa": "NYKAA.NS",
    # International — Taiwan / Semiconductor
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "mediatek": "MDTKF",
    "asus": "ASUUY",
    "acer": "ACEYY",
    # International — Australia / Other
    "commonwealth bank": "CMWAY",
    "westpac": "WBK",
    "anz": "ANZBY",
    "nab": "NABZY",
    "bhp billiton": "BHP",
    "woolworths": "WOLZY",
    "qantas": "QABSY",
}


@st.cache_data(ttl=3600)
def resolve_ticker(company_name: str) -> tuple[str | None, str | None]:
    """
    Resolve a company name to its ticker symbol.

    Returns:
        (ticker, company_full_name) or (None, None) if not found.
        Returns ("PRIVATE", company_name) if the company is known to be private.
    """
    if not company_name or not company_name.strip():
        return None, None

    query = company_name.strip().lower()

    # 0. Check if it's a known private company
    if query in PRIVATE_COMPANIES:
        return "PRIVATE", company_name.strip()

    # 1. Check alias map first (fast path — guaranteed correct ticker)
    if query in COMPANY_ALIASES:
        ticker = COMPANY_ALIASES[query]
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or ticker
            return ticker, name
        except Exception:
            return ticker, ticker

    # 2. If input looks like a ticker already (short, uppercase-ish), try direct
    if len(company_name.strip()) <= 6 and company_name.strip().replace(".", "").isalpha():
        ticker_upper = company_name.strip().upper()
        try:
            info = yf.Ticker(ticker_upper).info
            if info.get("regularMarketPrice") or info.get("currentPrice"):
                name = info.get("longName") or info.get("shortName") or ticker_upper
                return ticker_upper, name
        except Exception:
            pass

    # 3. Use Yahoo Finance search API with smart filtering
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": company_name,
            "quotesCount": 10,
            "newsCount": 0,
            "enableFuzzyQuery": True,
            "quotesQueryId": "tss_match_phrase_query",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        data = resp.json()
        quotes = data.get("quotes", [])

        # Score results: prefer major exchanges over subsidiaries/obscure listings
        PREFERRED_EXCHANGES = {
            "NMS", "NGM", "NCM",  # NASDAQ tiers
            "NYQ",                 # NYSE
            "NYSEArca",
            "PNK",                 # OTC Pink (ADRs land here)
            "GER",                 # Frankfurt (for European)
            "LSE",                 # London
            "NSE", "BSE",          # India
        }
        DEPRIORITIZED_SUFFIXES = (
            ".TW", ".TWO",         # Taiwan local (prefer ADR)
            ".HK",                 # Hong Kong local (prefer ADR)
            ".SS", ".SZ",          # China A-shares (prefer ADR/HK)
            ".KS", ".KQ",          # Korea local (prefer ADR)
        )

        equity_quotes = [q for q in quotes if q.get("quoteType") == "EQUITY"]

        if equity_quotes:
            # First pass: find one NOT on a deprioritized exchange suffix
            for q in equity_quotes:
                symbol = q.get("symbol", "")
                if not any(symbol.endswith(sfx) for sfx in DEPRIORITIZED_SUFFIXES):
                    name = q.get("longname") or q.get("shortname") or symbol
                    return symbol, name

            # Second pass: accept any equity
            q = equity_quotes[0]
            ticker = q.get("symbol")
            name = q.get("longname") or q.get("shortname") or ticker
            return ticker, name

        # Fallback to first result of any type
        if quotes:
            ticker = quotes[0].get("symbol")
            name = quotes[0].get("longname") or quotes[0].get("shortname") or ticker
            return ticker, name

    except Exception:
        pass

    return None, None


@st.cache_data(ttl=3600)
def get_company_info(ticker: str) -> dict:
    """Fetch full yfinance info dict for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info if info else {}
    except Exception:
        return {}
