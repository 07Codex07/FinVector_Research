import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/processed/final_news.csv")
df_prices = pd.read_csv("data/processed/aligned_prices.csv")
df_prices["Datetime"] = pd.to_datetime(df_prices["Datetime"]).dt.tz_convert("Asia/Kolkata")
df_prices = df_prices.sort_values("Datetime")

print(df_prices.shape)
print(df_prices.head())
print(df_prices.tail())
print(df.shape)
print(df.head())
print(df.tail())

# print(df_prices.describe())
# print(df_prices['Datetime'].diff().value_counts())
import mplfinance as mpf

df_plot = df_prices.copy()
df_plot.index = df_plot['Datetime']

# mpf.plot(df_plot, type='candle', style='yahoo', volume=False, title='NIFTY 30-Min OHLC')
df_prices['return'] = df_prices['Close'].pct_change()

# # Histogram of returns
# df_prices['return'].hist(bins=50, figsize=(8,4))

# # Rolling volatility
# df_prices['vol'] = df_prices['return'].rolling(10).std()
# df_prices['vol'].plot(figsize=(12,4))
df_prices['return'].describe()
df_prices['return'].skew()
df_prices['return'].kurt()

stats = {
    "Return Summary": df_prices['return'].describe(),
    "Skewness": df_prices['return'].skew(),
    "Kurtosis": df_prices['return'].kurt()
}

for key, value in stats.items():
    print(f"\n--- {key} ---")
    print(value)

# Plotting the distribution of returns
# df_prices['return'].hist(bins=50, figsize=(8,4))
# plt.title('Distribution of Returns')
# plt.xlabel('Return')
# plt.ylabel('Frequency')
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data (change path if needed)
df = pd.read_csv("data/processed/aligned_prices.csv")

# Compute returns if not already present
df["Return"] = df["Close"].pct_change()
returns = df["Return"].dropna()

# ============================
# 1. TAIL COMPARISON HISTOGRAM
# ============================

# plt.figure(figsize=(10,5))
# plt.hist(returns, bins=40)
# plt.title("Tail Comparison Histogram")
# plt.xlabel("Return")
# plt.ylabel("Frequency")

# # Mark the extremes
# plt.axvline(returns.min(), color='red', linewidth=2)
# plt.axvline(returns.max(), color='red', linewidth=2)

# plt.grid(True)
# plt.tight_layout()
# plt.show()



# # ============
# # 2. QQ PLOT
# # ============

# sm.qqplot(returns, line='s')
# plt.title("QQ Plot of Returns")
# plt.tight_layout()
# plt.show()
