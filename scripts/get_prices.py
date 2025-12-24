# ================================
# Download 30-minute NIFTY Data
# ================================

import yfinance as yf
import pandas as pd

def download_nifty_data():
    """
    Downloads 30-minute OHLCV data for NIFTY 50
    and saves it as a CSV file.
    """

    # NIFTY 50 Yahoo Finance ticker
    TICKER = "^NSEI"

    print("Downloading NIFTY price data...")

    # Download last 3 months of data at 30-minute interval
    data = yf.download(
       tickers=TICKER,
       period="60d",
       interval="30m",
       auto_adjust=True,
       progress=False,
)

    # Reset index so timestamp becomes a column
    data = data.reset_index()

    # Save to CSV
    SAVE_PATH = "data/raw_prices/nifty_30min.csv"
    data.to_csv(SAVE_PATH, index=False)

    print(f"✅ NIFTY data saved to: {SAVE_PATH}")


if __name__ == "__main__":
    download_nifty_data()
