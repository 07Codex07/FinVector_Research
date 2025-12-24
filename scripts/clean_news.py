# ===================================
# Convert UNIX timestamps to readable time
# ===================================

import pandas as pd

def clean_news_timestamps():
    """
    Converts UNIX timestamps from Yahoo Finance
    into readable datetime format.
    """

    INPUT_PATH = "data/raw_news/yahoo_nifty_news.csv"
    OUTPUT_PATH = "data/processed/cleaned_news.csv"

    print("🧹 Cleaning timestamps...")

    df = pd.read_csv(INPUT_PATH)

    # Convert UNIX time to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Cleaned news saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_news_timestamps()
