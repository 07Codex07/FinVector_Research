import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def scrape_et_markets(max_pages=200, days=60):
    print("➡️ Scraping ET Markets ({} days)...".format(days))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    base_url = "https://economictimes.indiatimes.com/markets/stocks/news"
    articles = []

    for page in range(1, max_pages + 1):
        print(f"   ➜ Page {page}")

        url = f"{base_url}?page={page}"
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        blocks = soup.select("div.eachStory")

        # If no blocks found → pagination ended
        if not blocks:
            print("   ❌ No more pages. Stopping.")
            break

        for block in blocks:
            headline_tag = block.find("a")
            time_tag = block.find("time")

            if not headline_tag or not time_tag:
                continue

            headline = headline_tag.text.strip()
            link = headline_tag["href"]

            if link.startswith("/"):
                link = "https://economictimes.indiatimes.com" + link

            # Parse timestamp (removing 'IST')
            raw_ts = time_tag.text.strip().replace(" IST", "")

            try:
                ts = pd.to_datetime(raw_ts)
            except:
                continue

            # STOP CONDITION  
            if ts < start_date:
                print("   ⛔ Reached older than {} days. Stopping.".format(days))
                return articles

            if ts > end_date:
                continue

            articles.append({
                "timestamp": ts,
                "headline": headline,
                "source": "ETMarkets",
                "url": link
            })

    return articles


def save_news(days=60):
    print("========== SCRAPING START ==========")

    data = scrape_et_markets(days=days)

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["headline"])
    df = df.sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)

    df.to_csv("data/processed/final_news.csv", index=False)

    print("===================================")
    print(f"✅ Total articles saved: {len(df)}")
    print("📁 File: data/processed/final_news.csv")


if __name__ == "__main__":
    save_news(days=60)
