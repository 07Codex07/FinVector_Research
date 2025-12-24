"""
all_visualizations.py

Generates:
1) Returns time series
2) Rolling mean & rolling volatility
3) Cumulative returns (wealth curve)
4) Histogram + KDE-like smooth overlay + Normal & Student-t fit overlay
5) ACF and PACF (if statsmodels available)
6) Bollinger Bands + Price with SMA/EMA
7) Drawdown chart
8) VaR (1%, 5%) and Expected Shortfall visualization
9) Correlation heatmap of features (uses matplotlib only)
10) Left-tail and right-tail QQ plots (statsmodels or scipy fallback)
11) Scatter plot with outlier annotation
12) Event-study around positive outliers with news markers
13) Rolling Sharpe and Rolling Beta (beta requires benchmark CSV path; optional)

Requirements:
- pandas, numpy, matplotlib
- (recommended) statsmodels, scipy

Run: python scripts/all_visualizations.py
"""

import os
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional libs
_use_statsmodels = False
_use_scipy = False
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf, pacf
    _use_statsmodels = True
except Exception:
    try:
        import scipy.stats as stats
        _use_scipy = True
    except Exception:
        pass

# -------------------------
# Config / Paths
# -------------------------
PRICE_PATH = "data/processed/aligned_prices.csv"
NEWS_PATH = "data/processed/final_news.csv"
BENCHMARK_PATH = None  # set to a CSV path with Datetime, Close if you want rolling beta
OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Utility functions
# -------------------------
def safe_read_prices(path):
    df = pd.read_csv(path)
    # normalize timestamp column
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["Datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        # try any column with 'time'
        for c in df.columns:
            if "time" in c.lower():
                df["Datetime"] = pd.to_datetime(df[c], utc=True, errors="coerce")
                break
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df

def compute_returns(df, price_col="Close", log_returns=False):
    if log_returns:
        returns = np.log(df[price_col]).diff()
    else:
        returns = df[price_col].pct_change()
    return returns

def draw_and_save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print("Saved:", path)

# -------------------------
# Load data
# -------------------------
prices = safe_read_prices(PRICE_PATH)
if "Close" not in prices.columns:
    raise ValueError("Price CSV must contain 'Close' column.")

# compute returns
prices["Return"] = compute_returns(prices, "Close", log_returns=False)
returns = prices["Return"].dropna()

# basic stats
mu = returns.mean()
sigma = returns.std(ddof=0)
skew = returns.skew()
excess_kurt = returns.kurt()

print(f"n={len(returns)} mean={mu:.6f} std={sigma:.6f} skew={skew:.4f} excess_kurtosis={excess_kurt:.4f}")

# -------------------------
# 1) Returns Time Series
# -------------------------
# --- FIXED Rolling Sharpe Plot ---


# -------------------------
# 2) Rolling mean & rolling volatility
# -------------------------
def rolling_metrics(series, window=20):
    roll_mean = series.rolling(window=window).mean()
    roll_vol = series.rolling(window=window).std()
    return roll_mean, roll_vol

for w in [5, 20, 60]:  
    rm, rv = rolling_metrics(returns, window=w)

    # Align by dropping the first NaN row
    # --- FIXED Rolling Sharpe Plot ---




# -------------------------
# 3) Cumulative returns (wealth curve)
# -------------------------
# Cumulative returns (wealth curve)
cum = (1 + returns).cumprod()
valid_idx = cum.index   # matches returns index (263 rows)

fig = plt.figure(figsize=(12,4))
plt.plot(prices.loc[valid_idx, "Datetime"], cum.loc[valid_idx], label="Wealth curve")

plt.title("Cumulative Returns (Wealth Curve)")
plt.xlabel("Datetime")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)

draw_and_save(fig, "03_cumulative_returns.png")
plt.close(fig)


# -------------------------
# 4) Histogram + Normal & Student-t overlay
# -------------------------
fig = plt.figure(figsize=(10,6))
n_bins = 50
counts, bins, _ = plt.hist(returns, bins=n_bins, alpha=0.75, density=False)
bin_centers = 0.5*(bins[:-1] + bins[1:])
# Normal PDF scaled
pdf_normal = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((bin_centers - mu)/sigma)**2)
bin_width = bins[1] - bins[0]
pdf_normal_scaled = pdf_normal * len(returns) * bin_width
plt.plot(bin_centers, pdf_normal_scaled, linewidth=2, linestyle='--', label='Normal (mu,std)')
# Student-t fit if scipy available
if _use_scipy:
    params = stats.t.fit(returns.dropna())
    df_t, loc_t, scale_t = params
    pdf_t = stats.t.pdf((bin_centers - loc_t)/scale_t, df_t) / scale_t
    pdf_t_scaled = pdf_t * len(returns) * bin_width
    plt.plot(bin_centers, pdf_t_scaled, linewidth=2, linestyle='-', label=f"Student-t df={df_t:.1f}")
plt.title("Histogram of Returns with Normal (and Student-t) overlay")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
draw_and_save(fig, "04_histogram_with_fits.png")
plt.close(fig)

# -------------------------
# 5) ACF & PACF
# -------------------------
if _use_statsmodels:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig = plt.figure(figsize=(10,4))
    plot_acf(returns, lags=40, ax=plt.gca())
    plt.title("Autocorrelation (ACF) of Returns")
    draw_and_save(fig, "05_acf_returns.png")
    plt.close(fig)

    fig = plt.figure(figsize=(10,4))
    plot_pacf(returns.fillna(0), lags=40, method='ywm', ax=plt.gca())
    plt.title("Partial Autocorrelation (PACF) of Returns")
    draw_and_save(fig, "06_pacf_returns.png")
    plt.close(fig)
else:
    print("statsmodels not available: skipping ACF/PACF plots. Install statsmodels for these.")

# -------------------------
# 6) Bollinger Bands + Price with SMA/EMA
# -------------------------
price_series = prices["Close"]
window = 20
sma = price_series.rolling(window=window).mean()
stddev = price_series.rolling(window=window).std()
upper = sma + 2*stddev
lower = sma - 2*stddev
ema = price_series.ewm(span=window, adjust=False).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(prices["Datetime"], price_series, label="Price")
plt.plot(prices["Datetime"], sma, label=f"SMA({window})")
plt.plot(prices["Datetime"], ema, label=f"EMA({window})")
plt.plot(prices["Datetime"], upper, label="Bollinger Upper")
plt.plot(prices["Datetime"], lower, label="Bollinger Lower")
plt.fill_between(prices["Datetime"], lower, upper, alpha=0.1)
plt.title("Price with SMA, EMA and Bollinger Bands")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
draw_and_save(fig, "07_price_sma_ema_bollinger.png")
plt.close(fig)

# -------------------------
# 7) Drawdown chart
# -------------------------
def compute_drawdown(wealth):
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return dd

dd = compute_drawdown(cum)
fig = plt.figure(figsize=(12,4))
# Align drawdown index with timestamps
valid_idx = dd.index

fig = plt.figure(figsize=(12,4))
plt.plot(prices.loc[valid_idx, "Datetime"], dd.loc[valid_idx], color="red")

plt.title("Drawdown Over Time")
plt.xlabel("Datetime")
plt.ylabel("Drawdown")
plt.grid(True, linestyle=":", alpha=0.5)

draw_and_save(fig, "08_drawdown.png")
plt.close(fig)

plt.title("Drawdown (from peak)")
plt.xlabel("Datetime")
plt.ylabel("Drawdown")
plt.grid(True, linestyle=":", alpha=0.5)
draw_and_save(fig, "08_drawdown.png")
plt.close(fig)

# -------------------------
# 8) VaR (1% & 5%) and ES (Expected Shortfall)
# -------------------------
def var_es(series, alpha=0.05):
    series = series.dropna()
    var = series.quantile(alpha)
    es = series[series <= var].mean()
    return var, es

var1, es1 = var_es(returns, alpha=0.01)
var5, es5 = var_es(returns, alpha=0.05)

fig = plt.figure(figsize=(10,5))
plt.hist(returns, bins=60, alpha=0.6)
plt.axvline(var1, color="red", linestyle="--", label=f"VaR 1% = {var1:.4f}")
plt.axvline(var5, color="orange", linestyle="--", label=f"VaR 5% = {var5:.4f}")
plt.title("VaR & ES visualization")
plt.legend()
draw_and_save(fig, "09_var_es.png")
plt.close(fig)

print(f"VaR 1%={var1:.4f}, ES 1%={es1:.4f}; VaR 5%={var5:.4f}, ES 5%={es5:.4f}")

# -------------------------
# 9) Correlation heatmap of selected features
# -------------------------
# Build feature dataframe: returns, lag1..lag5, rolling vol, SMA diff, EMA diff, etc.
feat = pd.DataFrame({"return": returns})
for lag in range(1,6):
    feat[f"lag_{lag}"] = returns.shift(lag)
feat["rolling_vol_20"] = returns.rolling(20).std()
feat["sma20_diff"] = (prices["Close"].rolling(20).mean() - prices["Close"]).shift(0)
feat["ema20_diff"] = (prices["Close"].ewm(span=20, adjust=False).mean() - prices["Close"]).shift(0)
feat = feat.dropna()

corr = feat.corr()

fig = plt.figure(figsize=(8,6))
plt.imshow(corr.values, interpolation='nearest')
plt.gca().set_xticks(range(len(corr.columns)))
plt.gca().set_yticks(range(len(corr.columns)))
plt.gca().set_xticklabels(corr.columns, rotation=45, ha='right')
plt.gca().set_yticklabels(corr.columns)
plt.colorbar()
plt.title("Correlation matrix (features)")
draw_and_save(fig, "10_correlation_matrix.png")
plt.close(fig)

# -------------------------
# 10) Left-tail and right-tail QQ plots (and full QQ)
# -------------------------
def qq_plot_using_statsmodels(series, title, fname):
    fig = plt.figure(figsize=(6,5))
    sm.qqplot(series.dropna(), line='s', ax=plt.gca())
    plt.title(title)
    draw_and_save(fig, fname)
    plt.close(fig)

def qq_plot_using_scipy(series, title, fname):
    import scipy.stats as sps
    fig = plt.figure(figsize=(6,5))
    sps.probplot(series.dropna(), dist="norm", plot=plt)
    plt.title(title)
    draw_and_save(fig, fname)
    plt.close(fig)

left_tail = returns[returns <= returns.quantile(0.05)]
right_tail = returns[returns >= returns.quantile(0.95)]

if _use_statsmodels:
    qq_plot_using_statsmodels(left_tail, "Left-tail QQ (<=5%)", "11_qq_left.png")
    qq_plot_using_statsmodels(right_tail, "Right-tail QQ (>=95%)", "12_qq_right.png")
    qq_plot_using_statsmodels(returns, "Full-sample QQ", "13_qq_full.png")
elif _use_scipy:
    qq_plot_using_scipy(left_tail, "Left-tail QQ (<=5%)", "11_qq_left.png")
    qq_plot_using_scipy(right_tail, "Right-tail QQ (>=95%)", "12_qq_right.png")
    qq_plot_using_scipy(returns, "Full-sample QQ", "13_qq_full.png")
else:
    print("Neither statsmodels nor scipy available for QQ plots. Install them to get QQ plots.")

# -------------------------
# 11) Scatter plot with outlier annotation
# -------------------------
threshold_pos = mu + 3*sigma
threshold_neg = mu - 3*sigma
out_pos = prices[prices["Return"] > threshold_pos]
out_neg = prices[prices["Return"] < threshold_neg]

fig = plt.figure(figsize=(12,4))
plt.scatter(prices["Datetime"], prices["Return"], s=8, alpha=0.5)
plt.scatter(out_pos["Datetime"], out_pos["Return"], color="red", label="positive outliers")
plt.scatter(out_neg["Datetime"], out_neg["Return"], color="purple", label="negative outliers")
plt.axhline(threshold_pos, color="red", linestyle="--")
plt.axhline(threshold_neg, color="purple", linestyle="--")
plt.title("Returns with outliers highlighted")
plt.xlabel("Datetime")
plt.ylabel("Return")
plt.legend()
# annotate top 5 positive outliers
for _, r in out_pos.sort_values("Return", ascending=False).head(5).iterrows():
    plt.annotate(r["Datetime"].strftime("%Y-%m-%d %H:%M"), (r["Datetime"], r["Return"]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
draw_and_save(fig, "14_returns_outliers.png")
plt.close(fig)

# -------------------------
# 12) Event-study: returns around positive outliers + news alignment
# -------------------------
# Load news if present
news_exists = os.path.exists(NEWS_PATH)
if news_exists:
    news = pd.read_csv(NEWS_PATH)
    # standardize news timestamp
    if "timestamp" in news.columns:
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    elif "Datetime" in news.columns:
        news["timestamp"] = pd.to_datetime(news["Datetime"], utc=True, errors="coerce")
    else:
        for c in news.columns:
            if "time" in c.lower():
                news["timestamp"] = pd.to_datetime(news[c], utc=True, errors="coerce")
                break
    news = news.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
else:
    news = None

window_hours = 3
window = timedelta(hours=window_hours)
event_series = []
event_times = []
for _, out in out_pos.iterrows():
    t = out["Datetime"]
    # slice returns within [-window, +window]
    mask = (prices["Datetime"] >= (t - window)) & (prices["Datetime"] <= (t + window))
    slice_df = prices.loc[mask, ["Datetime", "Return"]].set_index("Datetime")
    # reindex to uniform timeline around event (optional)
    # compute relative time in minutes
    rel = ((slice_df.index - t).total_seconds() / 60).astype(int)
    slice_df["rel_min"] = rel
    event_series.append(slice_df["Return"].values)
    event_times.append(t)

# Simple plotting: overlay event windows (aligned at t=0)
# We'll resample each slice on minute grid if needed, but here we simply plot raw points shifted by event index
fig = plt.figure(figsize=(10,6))
for i, t in enumerate(event_times):
    mask = (prices["Datetime"] >= (t - window)) & (prices["Datetime"] <= (t + window))
    s = prices.loc[mask, ["Datetime", "Return"]].copy()
    s["rel_min"] = ((s["Datetime"] - t).dt.total_seconds() / 60)
    plt.plot(s["rel_min"], s["Return"], alpha=0.6)
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Event windows of returns around positive outliers (±{window_hours} hours)")
plt.xlabel("Minutes relative to outlier")
plt.ylabel("Return")
draw_and_save(fig, "15_event_windows_pos_outliers.png")
plt.close(fig)

# Align news to each positive outlier and save matches
OUT_MATCHES = "outputs/outlier_news_matches.csv"
matches = []
if news is not None and not out_pos.empty:
    for _, out in out_pos.iterrows():
        t = out["Datetime"]
        mask = (news["timestamp"] >= (t - window)) & (news["timestamp"] <= (t + window))
        matched = news.loc[mask]
        if matched.empty:
            # expand to same day
            dm = news[news["timestamp"].dt.date == t.date()]
            matched = dm
        if matched.empty:
            matches.append({
                "outlier_time": t.isoformat(),
                "return": float(out["Return"]),
                "news_time": None,
                "headline": None,
                "source": None,
                "url": None
            })
        else:
            for _, n in matched.iterrows():
                matches.append({
                    "outlier_time": t.isoformat(),
                    "return": float(out["Return"]),
                    "news_time": n["timestamp"].isoformat(),
                    "headline": n.get("headline", ""),
                    "source": n.get("source", ""),
                    "url": n.get("url", "")
                })
    pd.DataFrame(matches).to_csv(OUT_MATCHES, index=False)
    print("Saved outlier-news matches to:", OUT_MATCHES)
else:
    print("No news file or no positive outliers; skipping news alignment.")

# -------------------------
# 13) Rolling Sharpe and (optional) Rolling Beta
# -------------------------
# Rolling Sharpe (annualization not applied here; adjust for interval frequency)
window = 20
rolling_mean = returns.rolling(window=window).mean()
rolling_std = returns.rolling(window=window).std()
rolling_sharpe = (rolling_mean / rolling_std).dropna()

# --- CORRECT Rolling Sharpe Plot ---
valid_idx = rolling_sharpe.index

fig = plt.figure(figsize=(12,4))
plt.plot(prices.loc[valid_idx, "Datetime"], rolling_sharpe, label="Sharpe")

plt.title(f"Rolling Sharpe (window={window})")
plt.xlabel("Datetime")
plt.ylabel("Sharpe")
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend()

draw_and_save(fig, "16_rolling_sharpe.png")
plt.close(fig)

# Rolling Beta if benchmark provided
if BENCHMARK_PATH and os.path.exists(BENCHMARK_PATH):
    bench = safe_read_prices(BENCHMARK_PATH)
    bench["Return"] = compute_returns(bench, "Close", log_returns=False)
    merged = pd.merge_asof(prices[["Datetime","Return"]].dropna(), bench[["Datetime","Return"]].dropna(),
                           on="Datetime", suffixes=("", "_bench"))
    # rolling beta via covariance / var
    rolling_cov = merged["Return"].rolling(window=window).cov(merged["Return_bench"])
    rolling_var = merged["Return_bench"].rolling(window=window).var()
    rolling_beta = (rolling_cov / rolling_var).fillna(0)
    fig = plt.figure(figsize=(12,4))
    plt.plot(merged["Datetime"], rolling_beta)
    plt.title(f"Rolling Beta to benchmark (window={window})")
    draw_and_save(fig, "17_rolling_beta.png")
    plt.close(fig)
else:
    print("No benchmark provided or file missing; skipping rolling beta.")

print("All visualizations generated and saved to", OUT_DIR)
