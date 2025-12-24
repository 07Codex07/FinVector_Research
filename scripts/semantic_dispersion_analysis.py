# ============================================
# Semantic Dispersion → Volatility Analysis
# ============================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm

# -------------------------------
# CONFIG
# -------------------------------
NEWS_PATH = "data/processed/final_news.csv"
EMB_PATH  = "data/embeddings/embeddings_finbert.npy"
PRICE_PATH = "data/processed/aligned_prices.csv"

WINDOW = "30min"

# -------------------------------
# LOAD DATA
# -------------------------------
print("📥 Loading data...")

news = pd.read_csv(NEWS_PATH)
news["timestamp"] = pd.to_datetime(news["timestamp"]).dt.tz_localize(None)

embeddings = np.load(EMB_PATH)

prices = pd.read_csv(PRICE_PATH)
prices["Datetime"] = pd.to_datetime(prices["Datetime"], utc=True).dt.tz_convert(None)

print(f"   News articles: {len(news)}")
print(f"   Embedding shape: {embeddings.shape}")
print(f"   Price rows: {len(prices)}")

# -------------------------------
# ASSIGN TIME WINDOWS
# -------------------------------
news["window"] = news["timestamp"].dt.floor(WINDOW)

# -------------------------------
# SEMANTIC DISPERSION FUNCTION
# -------------------------------
def semantic_dispersion(vectors):
    """
    Mean cosine distance from centroid
    """
    centroid = vectors.mean(axis=0, keepdims=True)
    distances = cosine_distances(vectors, centroid)
    return distances.mean()

# -------------------------------
# COMPUTE DISPERSION PER WINDOW
# -------------------------------
print("🧮 Computing semantic dispersion...")

dispersion_records = []

for window, group in news.groupby("window"):
    idx = group.index.values

    if len(idx) < 3:
        continue  # skip sparse windows

    vecs = embeddings[idx]
    disp = semantic_dispersion(vecs)

    dispersion_records.append({
        "window": window,
        "semantic_dispersion": disp,
        "news_count": len(idx)
    })

disp_df = pd.DataFrame(dispersion_records)
print(f"   Windows computed: {len(disp_df)}")

# -------------------------------
# PRICE → VOLATILITY
# -------------------------------
print("📉 Computing volatility...")

prices = prices.sort_values("Datetime")
prices["return"] = np.log(prices["Close"]).diff()
prices["volatility"] = prices["return"].abs()

prices["window"] = prices["Datetime"]

price_df = prices[["window", "volatility"]].dropna()

# -------------------------------
# MERGE NEWS + PRICE
# -------------------------------
df = pd.merge(
    disp_df,
    price_df,
    on="window",
    how="inner"
)

# Predict NEXT-window volatility
df["next_volatility"] = df["volatility"].shift(-1)
df = df.dropna()

print(f"   Final merged rows: {len(df)}")

# -------------------------------
# STATISTICAL TESTS
# -------------------------------
print("\n📊 Statistical Tests")

pearson_corr, p1 = pearsonr(df["semantic_dispersion"], df["next_volatility"])
spearman_corr, p2 = spearmanr(df["semantic_dispersion"], df["next_volatility"])

print(f"Pearson r   : {pearson_corr:.3f} (p={p1:.4f})")
print(f"Spearman ρ : {spearman_corr:.3f} (p={p2:.4f})")

# -------------------------------
# REGRESSION (CONTROL FOR NEWS COUNT)
# -------------------------------
X = df[["semantic_dispersion", "news_count"]]
X = sm.add_constant(X)
y = df["next_volatility"]

model = sm.OLS(y, X).fit()

print("\n📈 Regression Summary")
print(model.summary())

# -------------------------------
# SAVE OUTPUTS
# -------------------------------
df.to_csv("outputs/semantic_dispersion_features.csv", index=False)

with open("outputs/semantic_dispersion_results.txt", "w") as f:
    f.write(f"Pearson r: {pearson_corr:.4f} (p={p1:.4f})\n")
    f.write(f"Spearman rho: {spearman_corr:.4f} (p={p2:.4f})\n")
    f.write(model.summary().as_text())

print("\n✅ Analysis complete.")
print("📁 Saved:")
print("   - outputs/semantic_dispersion_features.csv")
print("   - outputs/semantic_dispersion_results.txt")
