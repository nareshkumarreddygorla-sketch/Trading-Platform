"""
Download historical NSE stock data for AI model training.
Uses yfinance to fetch OHLCV data for top NSE stocks.
Saves to data/nse_historical/ as parquet files.

Usage:
    PYTHONPATH=. python scripts/download_nse_data.py
    PYTHONPATH=. python scripts/download_nse_data.py --symbols RELIANCE,INFY,TCS --period 2y
"""

import argparse
import logging
import os

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Top NSE stocks by market cap
TOP_NSE_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "KOTAKBANK",
    "LT",
    "AXISBANK",
    "BAJFINANCE",
    "ASIANPAINT",
    "MARUTI",
    "TITAN",
    "SUNPHARMA",
    "ULTRACEMCO",
    "WIPRO",
    "HCLTECH",
    "NESTLEIND",
    "BAJAJFINSV",
    "ONGC",
    "NTPC",
    "POWERGRID",
    "ADANIENT",
    "ADANIPORTS",
    "JSWSTEEL",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "INDUSINDBK",
    "HINDALCO",
    "COALINDIA",
    "DRREDDY",
    "DIVISLAB",
    "GRASIM",
    "CIPLA",
    "EICHERMOT",
    "APOLLOHOSP",
    "HEROMOTOCO",
    "BPCL",
    "BRITANNIA",
    "TATACONSUM",
    "SBILIFE",
    "HDFCLIFE",
    "M&M",
    "UPL",
    "BAJAJ-AUTO",
    "SHREECEM",
]

# Nifty 50 index
NIFTY_SYMBOL = "^NSEI"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nse_historical")


def download_stock(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a single NSE stock."""
    yf_symbol = f"{symbol}.NS" if not symbol.startswith("^") else symbol
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning("No data for %s", symbol)
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "timestamp"
        df["symbol"] = symbol
        logger.info(
            "Downloaded %s: %d bars (%s to %s)",
            symbol,
            len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        return df
    except Exception as e:
        logger.error("Failed to download %s: %s", symbol, e)
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download NSE historical data")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: top 50 NSE)")
    parser.add_argument("--period", type=str, default="2y", help="Data period: 1y, 2y, 5y, max (default: 2y)")
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval: 1m, 5m, 1h, 1d (default: 1d)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else TOP_NSE_SYMBOLS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    success = 0

    # Download Nifty 50 index
    logger.info("Downloading Nifty 50 index...")
    nifty_df = download_stock(NIFTY_SYMBOL, period=args.period, interval=args.interval)
    if not nifty_df.empty:
        nifty_path = os.path.join(OUTPUT_DIR, "NIFTY50.parquet")
        nifty_df.to_parquet(nifty_path)
        logger.info("Saved Nifty 50 to %s", nifty_path)

    # Download individual stocks
    logger.info("Downloading %d stocks...", len(symbols))
    for symbol in symbols:
        df = download_stock(symbol, period=args.period, interval=args.interval)
        if not df.empty:
            # Save individual stock
            stock_path = os.path.join(OUTPUT_DIR, f"{symbol}.parquet")
            df.to_parquet(stock_path)
            all_data.append(df)
            success += 1

    # Save combined dataset
    if all_data:
        combined = pd.concat(all_data, ignore_index=False)
        combined_path = os.path.join(OUTPUT_DIR, "all_stocks.parquet")
        combined.to_parquet(combined_path)
        logger.info("Combined dataset: %d rows, saved to %s", len(combined), combined_path)

    logger.info("Download complete: %d/%d stocks successful", success, len(symbols))
    logger.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
