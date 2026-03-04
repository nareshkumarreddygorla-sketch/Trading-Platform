"""
NSE market universe: fetch NIFTY 500 constituents for full-market scanning.
Caches to disk (24h TTL). Provides yfinance-compatible tickers (.NS suffix).
Bundled fallback of top 200 liquid NSE stocks if network fetch fails.
"""
import csv
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "cache"
CACHE_FILE = CACHE_DIR / "nse_universe.json"
CACHE_TTL_SECONDS = 86400  # 24 hours

# Wikipedia page for NIFTY 500 company list — reliable and scrapeable
NIFTY500_URL = "https://en.wikipedia.org/wiki/NIFTY_500"


# ---------------------------------------------------------------------------
# Bundled fallback: top 200 liquid NSE stocks (always available offline)
# ---------------------------------------------------------------------------
FALLBACK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "TITAN",
    "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND", "BAJAJFINSV",
    "TATAMOTORS", "TECHM", "NTPC", "POWERGRID", "M&M", "ONGC",
    "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "GRASIM", "DIVISLAB", "DRREDDY", "CIPLA", "HDFCLIFE", "SBILIFE",
    "BPCL", "EICHERMOT", "HEROMOTOCO", "APOLLOHOSP", "BAJAJ-AUTO",
    "BRITANNIA", "INDUSINDBK", "TATACONSUM", "UPL", "DABUR",
    "GODREJCP", "HAVELLS", "PIDILITIND", "MARICO", "BERGEPAINT",
    "SIEMENS", "AMBUJACEM", "ACC", "SHREECEM", "MCDOWELL-N",
    "BIOCON", "LUPIN", "TORNTPHARM", "AUROPHARMA", "ALKEM",
    "IPCALAB", "MINDTREE", "MPHASIS", "LTTS", "PERSISTENT",
    "COFORGE", "NAUKRI", "IRCTC", "JUBLFOOD", "TRENT",
    "PIIND", "CHOLAFIN", "MUTHOOTFIN", "BAJAJHLDNG", "VOLTAS",
    "TVSMOTOR", "ASHOKLEY", "BALKRISIND", "MRF", "MOTHERSON",
    "PAGEIND", "COLPAL", "HINDPETRO", "IOC", "GAIL",
    "PETRONET", "IGL", "MGL", "CONCOR", "DLF",
    "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "PHOENIXLTD", "SUNTV",
    "ZEEL", "PVR", "FEDERALBNK", "BANDHANBNK", "IDFCFIRSTB",
    "RBLBANK", "CUB", "MANAPPURAM", "L&TFH", "PEL",
    "CANFINHOME", "RECLTD", "PFC", "IRFC", "NHPC",
    "SJVN", "CESC", "TATAPOWER", "ADANIGREEN", "ADANITRANS",
    "TORNTPOWER", "JSL", "JINDALSTEL", "SAIL", "NMDC",
    "VEDL", "NATIONALUM", "HINDALCO", "HINDCOPPER", "APLAPOLLO",
    "RATNAMANI", "KPITTECH", "HAPPSTMNDS", "ZOMATO", "NYKAA",
    "PAYTM", "POLICYBZR", "DELHIVERY", "STARHEALTH", "LICI",
    "SBICARD", "ICICIPRULI", "HDFCAMC", "ICICIGI", "NIACL",
    "INDUSTOWER", "BHARTIHEXA", "IDEA", "MFSL", "MAXHEALTH",
    "FORTIS", "METROPOLIS", "LALPATHLAB", "ABCAPITAL", "ATUL",
    "DEEPAKNTR", "AARTI", "CLEAN", "SRF", "FLUOROCHEM",
    "NAVINFLUOR", "SUMICHEM", "UBL", "RAJESHEXPO", "VBL",
    "CROMPTON", "WHIRLPOOL", "BLUESTARCO", "KAJARIACER", "CENTURYTEX",
    "ABFRL", "RAYMOND", "RELAXO", "BATAINDIA", "CAMPUS",
    "DIXON", "AMBER", "POLYCAB", "KEI", "CUMMINSIND",
    "THERMAX", "BEL", "HAL", "BDL", "GRINDWELL",
]


def _fetch_nifty500_from_web() -> List[str]:
    """Try to scrape NIFTY 500 symbols from Wikipedia."""
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(NIFTY500_URL, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the table with company symbols
        symbols = []
        for table in soup.find_all("table", {"class": "wikitable"}):
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header
                cols = row.find_all("td")
                if len(cols) >= 2:
                    # Symbol is typically in the first or second column
                    sym_text = cols[1].get_text(strip=True)
                    if sym_text and sym_text.isalpha():
                        symbols.append(sym_text.upper())
                    elif cols[0].get_text(strip=True).isalpha():
                        symbols.append(cols[0].get_text(strip=True).upper())
        if len(symbols) >= 100:
            return symbols
    except Exception as e:
        logger.debug("Wikipedia NIFTY 500 fetch failed: %s", e)

    # Alternative: fetch from NSE India indices CSV (public endpoint)
    try:
        import requests
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        symbols = []
        for row in reader:
            sym = row.get("Symbol", "").strip()
            if sym:
                symbols.append(sym.upper())
        if symbols:
            return symbols
    except Exception as e:
        logger.debug("NSE India CSV fetch failed: %s", e)

    return []


def _load_cache() -> List[str]:
    """Load cached universe if fresh enough."""
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
            if time.time() - data.get("ts", 0) < CACHE_TTL_SECONDS:
                return data["symbols"]
    except Exception:
        pass
    return []


def _save_cache(symbols: List[str]) -> None:
    """Persist universe to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({"ts": time.time(), "symbols": symbols}))
    except Exception as e:
        logger.warning("Cache save failed: %s", e)


def get_universe(*, yfinance_suffix: bool = True, force_refresh: bool = False) -> List[str]:
    """
    Return the full NSE trading universe (NIFTY 500).

    Args:
        yfinance_suffix: if True, append '.NS' to each symbol for yfinance compatibility
        force_refresh: bypass cache and re-fetch from web

    Returns:
        List of stock symbols (e.g. ['RELIANCE.NS', 'INFY.NS', ...])
    """
    symbols = [] if force_refresh else _load_cache()

    if not symbols:
        symbols = _fetch_nifty500_from_web()
        if symbols:
            _save_cache(symbols)
            logger.info("Fetched %d symbols from web for NSE universe", len(symbols))
        else:
            symbols = list(FALLBACK_SYMBOLS)
            logger.info("Using fallback universe (%d symbols)", len(symbols))

    # Deduplicate
    seen = set()
    unique = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    if yfinance_suffix:
        return [f"{s}.NS" for s in unique]
    return unique


def get_universe_raw() -> List[str]:
    """Return raw symbols without .NS suffix."""
    return get_universe(yfinance_suffix=False)
