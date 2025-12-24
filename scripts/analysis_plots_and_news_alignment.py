"""
analysis_plots_and_news_alignment.py

Requirements:
- pandas, numpy, matplotlib
- (recommended) statsmodels
- (fallback) scipy

Run: python scripts/analysis_plots_and_news_alignment.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Try imports for QQ plotting
_use_statsmodels = False
_use_scipy = False
try:
    import statsmodels.api as sm
    _use_statsmodels = True
except Exception:
    try:
        from scipy import stats
        _use_scipy = True
    except Exception:
        raise ImportError(
            "Please install statsmodels (`pip install statsmodels`) or scipy (`pip install scipy`) "
            "to run QQ plots. statsmodels is recommended."
        )

# -----------------------
# 1) Load price data
# -----------------------
PRICE_PATH = "data/processed/aligned_prices.csv"   # change if needed
NEWS_PATH = "data/processed/final_news.csv"        # change if needed
OUT_MATCHES = "outlier_news_matches.csv"

df = pd.read_csv(PRICE_PATH)
# Ensure Datetime parsed
if "Datetime" in df.columns:
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
elif "timestamp" in df.columns:
    df["Datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
else:
    raise ValueError("Couldn't find a timestamp column in price CSV. Expected 'Datetime' or 'timestamp'.")

# Compute simple pct returns (change to log returns if you prefer)
df = df.sort_values("Datetime").reset_index(drop=True)
df["Return"] = df["Close"].pct_change()
returns = df["Return"].dropna()

# Basic stats
mu = returns.mean()
sigma = returns.std(ddof=0)    # population std to match normal pdf overlay
skew = returns.skew()
kurtosis = returns.kurt()      # pandas gives "excess kurtosis" by default (kurtosis of sample - 3)
# If you want "raw" kurtosis (not excess): use scipy.stats.kurtosis(..., fisher=False)

print(f"n={len(returns):d} mean={mu:.6f} std={sigma:.6f} skew={skew:.6f} excess_kurtosis={kurtosis:.6f}")

# -----------------------
# 2) Combined skew/kurtosis interpretation + distribution + normal curve overlay
# -----------------------
plt.figure(figsize=(10,6))
n_bins = 50
counts, bins, patches = plt.hist(returns, bins=n_bins, density=False, alpha=0.75, color="#1f77b4", edgecolor="white")

# Overlay normal curve scaled to histogram counts
# Compute PDF values at bin centers then scale by total counts*bin_width to match histogram scale
bin_centers = 0.5*(bins[:-1] + bins[1:])
pdf_vals = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((bin_centers - mu)/sigma)**2)
# scale to histogram counts
bin_width = bins[1] - bins[0]
pdf_scaled = pdf_vals * len(returns) * bin_width
plt.plot(bin_centers, pdf_scaled, color="red", lw=2, label="Normal PDF (mean,std)")

# annotate skew/kurtosis and percentiles
plt.axvline(mu, color="black", linestyle="--", lw=1)
for q, c in zip([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                ["#444444", "#666666", "#888888", "#000000", "#888888", "#666666", "#444444"]):
    plt.axvline(returns.quantile(q), color=c, linestyle=":", lw=0.8, alpha=0.8)

plt.title("Distribution of Returns (hist) — Normal PDF overlay")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
txt = f"count={len(returns)}\nmean={mu:.6f}\nstd={sigma:.6f}\nskew={skew:.4f}\nexcess_kurtosis={kurtosis:.4f}"
plt.gcf().text(0.75, 0.6, txt, bbox=dict(facecolor="white", alpha=0.8), fontsize=9)

plt.tight_layout()
plt.show()

# -----------------------
# 3) Left-tail and Right-tail QQ plots
#    We'll use central quantile thresholds to isolate tails
# -----------------------
# Define tail thresholds
left_q = 0.05
right_q = 0.95

left_tail = returns[returns <= returns.quantile(left_q)]
right_tail = returns[returns >= returns.quantile(right_q)]

def plot_qq(data, title):
    plt.figure(figsize=(6,5))
    if _use_statsmodels:
        sm.qqplot(data, line='s', ax=plt.gca())
        plt.title(title + " (vs Normal) - statsmodels")
    else:
        # fallback using scipy.stats.probplot
        import scipy.stats as sps
        sps.probplot(data, dist="norm", plot=plt)
        plt.title(title + " (vs Normal) - scipy")
    plt.tight_layout()
    plt.show()

plot_qq(left_tail, f"Left-tail QQ plot (<= {int(left_q*100)}th pct)")
plot_qq(right_tail, f"Right-tail QQ plot (>= {int(right_q*100)}th pct)")

# Full QQ for reference
plot_qq(returns, "Full-sample QQ plot (for reference)")

# -----------------------
# 4) Scatter plot of returns vs time with outliers annotated
#    We'll define positive outliers as returns > mu + 3*sigma (adjustable)
# -----------------------
threshold_pos = mu + 3*sigma
threshold_neg = mu - 3*sigma

outliers_pos = df[df["Return"] > threshold_pos]
outliers_neg = df[df["Return"] < threshold_neg]

plt.figure(figsize=(12,5))
plt.plot(df["Datetime"], df["Return"], marker=".", linestyle="none", alpha=0.6, label="returns")
plt.scatter(outliers_pos["Datetime"], outliers_pos["Return"], color="red", label=f"pos outliers > {threshold_pos:.4f}")
plt.scatter(outliers_neg["Datetime"], outliers_neg["Return"], color="purple", label=f"neg outliers < {threshold_neg:.4f}")
plt.axhline(threshold_pos, color="red", linestyle="--", alpha=0.7)
plt.axhline(threshold_neg, color="purple", linestyle="--", alpha=0.7)
plt.title("Returns over time (outliers highlighted)")
plt.xlabel("Datetime")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()

# annotate top positive outliers with timestamps (avoid clutter)
for _, r in outliers_pos.sort_values("Return", ascending=False).head(10).iterrows():
    plt.annotate(r["Datetime"].strftime("%Y-%m-%d %H:%M"), (r["Datetime"], r["Return"]),
                 textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color="red")

plt.show()

# -----------------------
# 5) News alignment for positive outliers
#    For each positive outlier, find news within +/- 3 hours (adjustable)
# -----------------------
if os.path.exists(NEWS_PATH):
    news = pd.read_csv(NEWS_PATH)
    # standardize timestamp column names in news
    if "timestamp" in news.columns:
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    elif "Datetime" in news.columns:
        news["timestamp"] = pd.to_datetime(news["Datetime"], utc=True, errors="coerce")
    else:
        # try common names
        possible = [c for c in news.columns if "time" in c.lower()]
        if possible:
            news["timestamp"] = pd.to_datetime(news[possible[0]], utc=True, errors="coerce")
        else:
            raise ValueError("Could not find a timestamp column in news CSV.")

    news = news.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # ensure timezone awareness consistent: our df Datetime was parsed with utc=True
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")

    matches = []
    window = timedelta(hours=3)   # +/- 3 hours
    for _, out in outliers_pos.iterrows():
        out_time = out["Datetime"]
        # find news within window
        mask = (news["timestamp"] >= (out_time - window)) & (news["timestamp"] <= (out_time + window))
        matched = news[mask]
        if len(matched) == 0:
            # if no news in 3 hours, optionally expand to same day
            daymask = (news["timestamp"].dt.date == out_time.date())
            matched = news[daymask]
        if len(matched) == 0:
            # no match found
            matches.append({
                "outlier_time": out_time.isoformat(),
                "return": out["Return"],
                "news_time": None,
                "news_source": None,
                "news_headline": None,
                "news_url": None
            })
        else:
            # add each matched news row
            for _, n in matched.iterrows():
                matches.append({
                    "outlier_time": out_time.isoformat(),
                    "return": out["Return"],
                    "news_time": n["timestamp"].isoformat(),
                    "news_source": n.get("source", None),
                    "news_headline": n.get("headline", None),
                    "news_url": n.get("url", None)
                })
    # Save matches
    if len(matches) > 0:
        df_matches = pd.DataFrame(matches)
        df_matches.to_csv(OUT_MATCHES, index=False)
        print(f"Saved {len(df_matches)} outlier-news matches to: {OUT_MATCHES}")
    else:
        print("No positive outliers found or no news matches.")
else:
    print(f"News file not found at {NEWS_PATH}. Skipping news alignment step.")

# -----------------------
# End
# -----------------------
print("Done.")
