import pandas as pd

def align_prices():
    print("📉 Aligning price data with news time range...")

    NEWS_PATH = "data/processed/final_news.csv"
    PRICE_PATH = "data/raw_prices/nifty_30min.csv"
    OUTPUT_PATH = "data/processed/aligned_prices.csv"

    news = pd.read_csv(NEWS_PATH)
    prices = pd.read_csv(PRICE_PATH)

    # Parse timestamps as UTC
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
    prices["Datetime"] = pd.to_datetime(prices["Datetime"], utc=True)

    # Align to news time range
    start_date = news["timestamp"].min()
    end_date   = news["timestamp"].max()

    print("Filtering prices between:", start_date, "and", end_date)
    print("Price rows before filtering:", len(prices))

    filtered_prices = prices[
        (prices["Datetime"] >= start_date) &
        (prices["Datetime"] <= end_date)
    ]

    print("Price rows after filtering:", len(filtered_prices))

    filtered_prices.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Aligned price data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    align_prices()
