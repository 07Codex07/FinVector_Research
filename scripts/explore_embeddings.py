"""
Exploratory analysis of news embeddings
Visualize high-dimensional data and understand structure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def create_output_dir():
    os.makedirs("outputs/exploration", exist_ok=True)

def analyze_pca_variance():
    """
    Analyze how much variance is captured by principal components
    This tells us about the information structure in embeddings
    """
    
    print("=" * 60)
    print("📊 PCA VARIANCE ANALYSIS")
    print("=" * 60)
    
    # Load data
    embeddings = np.load("data/embeddings/embeddings_finbert.npy")
    print(f"\nOriginal embedding shape: {embeddings.shape}")
    
    # Perform PCA with 50 components
    print("\n🔄 Computing PCA (50 components)...")
    pca = PCA(n_components=50)
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Save PCA model for later use
    import joblib
    joblib.dump(pca, "data/embeddings/pca_model.pkl")
    
    # Calculate cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Print key statistics
    print("\n📈 Variance Statistics:")
    print(f"   First 2 PCs: {cumvar[1]:.2%}")
    print(f"   First 5 PCs: {cumvar[4]:.2%}")
    print(f"   First 10 PCs: {cumvar[9]:.2%}")
    print(f"   First 20 PCs: {cumvar[19]:.2%}")
    print(f"   First 50 PCs: {cumvar[49]:.2%}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Cumulative variance
    axes[0].plot(range(1, 51), cumvar, 'b-', linewidth=2)
    axes[0].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    axes[0].axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
    axes[0].set_xlabel('Number of Principal Components', fontsize=12)
    axes[0].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[0].set_title('PCA Cumulative Variance', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Individual component variance
    axes[1].bar(range(1, 21), pca.explained_variance_ratio_[:20], color='steelblue')
    axes[1].set_xlabel('Principal Component', fontsize=12)
    axes[1].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[1].set_title('Variance per PC (First 20)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("outputs/exploration/pca_variance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: outputs/exploration/pca_variance_analysis.png")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if cumvar[1] > 0.15:
        print("   ✓ First 2 PCs capture >15% variance → Good 2D visualization possible")
    else:
        print("   ⚠ First 2 PCs capture <15% variance → Data is highly distributed")
    
    if cumvar[9] > 0.50:
        print("   ✓ First 10 PCs capture >50% variance → Clear structure exists")
    else:
        print("   ⚠ First 10 PCs capture <50% variance → Diffuse semantic space")
    
    return pca, pca_embeddings


def visualize_2d_embedding(pca, show_density=True):
    """
    Create 2D visualization of embeddings using PCA
    """
    
    print("\n" + "=" * 60)
    print("🎨 2D VISUALIZATION")
    print("=" * 60)
    
    # Load data
    embeddings = np.load("data/embeddings/embeddings_finbert.npy")
    news = pd.read_csv("data/processed/final_news.csv")
    
    # Transform to 2D
    print("\n🔄 Projecting to 2D space...")
    coords_2d = pca.transform(embeddings)[:, :2]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Simple scatter
    axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                   alpha=0.4, s=20, c='steelblue', edgecolors='none')
    axes[0].set_xlabel('First Principal Component', fontsize=12)
    axes[0].set_ylabel('Second Principal Component', fontsize=12)
    axes[0].set_title('News Embeddings in 2D Space (Unlabeled)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Density plot
    if show_density:
        axes[1].hexbin(coords_2d[:, 0], coords_2d[:, 1], 
                      gridsize=30, cmap='YlOrRd', mincnt=1)
        axes[1].set_xlabel('First Principal Component', fontsize=12)
        axes[1].set_ylabel('Second Principal Component', fontsize=12)
        axes[1].set_title('Density Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(axes[1].collections[0], ax=axes[1], label='Article Count')
    
    plt.tight_layout()
    plt.savefig("outputs/exploration/embeddings_2d_unlabeled.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: outputs/exploration/embeddings_2d_unlabeled.png")
    
    # Check for visible structure
    print("\n🔍 Visual Inspection:")
    print("   Look at the scatter plot:")
    print("   - Do you see natural groupings/clusters?")
    print("   - Are there outliers (far from main group)?")
    print("   - Is data spread evenly or concentrated?")
    
    return coords_2d


def temporal_analysis(coords_2d):
    """
    Analyze how embeddings change over time
    """
    
    print("\n" + "=" * 60)
    print("⏰ TEMPORAL ANALYSIS")
    print("=" * 60)
    
    news = pd.read_csv("data/processed/final_news.csv")
    news['timestamp'] = pd.to_datetime(news['timestamp'])
    news['date'] = news['timestamp'].dt.date
    
    # Color by time
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create time-based colormap
    dates = news['timestamp']
    normalize = plt.Normalize(vmin=dates.min().value, vmax=dates.max().value)
    
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        c=dates.values.astype(np.int64), 
                        cmap='viridis', alpha=0.6, s=30,
                        norm=normalize)
    
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title('Embeddings Colored by Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar with dates
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("outputs/exploration/embeddings_temporal.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: outputs/exploration/embeddings_temporal.png")
    
    print("\n💡 What to look for:")
    print("   - Do recent articles cluster differently than older ones?")
    print("   - Are there temporal shifts in the embedding space?")


def sample_headlines_by_location(coords_2d):
    """
    Show sample headlines from different regions of the 2D space
    """
    
    print("\n" + "=" * 60)
    print("📝 SAMPLE HEADLINES BY REGION")
    print("=" * 60)
    
    news = pd.read_csv("data/processed/final_news.csv")
    
    # Define regions (quadrants)
    pc1_median = np.median(coords_2d[:, 0])
    pc2_median = np.median(coords_2d[:, 1])
    
    regions = {
        'Top-Right (PC1+, PC2+)': (coords_2d[:, 0] > pc1_median) & (coords_2d[:, 1] > pc2_median),
        'Top-Left (PC1-, PC2+)': (coords_2d[:, 0] < pc1_median) & (coords_2d[:, 1] > pc2_median),
        'Bottom-Right (PC1+, PC2-)': (coords_2d[:, 0] > pc1_median) & (coords_2d[:, 1] < pc2_median),
        'Bottom-Left (PC1-, PC2-)': (coords_2d[:, 0] < pc1_median) & (coords_2d[:, 1] < pc2_median),
    }
    
    for region_name, mask in regions.items():
        print(f"\n🗺️ {region_name}")
        print(f"   {mask.sum()} articles")
        print("   Sample headlines:")
        samples = news[mask].sample(min(3, mask.sum()))
        for _, row in samples.iterrows():
            print(f"   • {row['headline'][:80]}...")


def main():
    """
    Run all exploratory analyses
    """
    
    create_output_dir()
    
    # 1. PCA variance analysis
    pca, pca_embeddings = analyze_pca_variance()
    
    # 2. 2D visualization
    coords_2d = visualize_2d_embedding(pca)
    
    # 3. Temporal analysis
    temporal_analysis(coords_2d)
    
    # 4. Sample headlines
    sample_headlines_by_location(coords_2d)
    
    print("\n" + "=" * 60)
    print("✅ EXPLORATION COMPLETE")
    print("=" * 60)
    print("\n📊 Generated visualizations:")
    print("   1. outputs/exploration/pca_variance_analysis.png")
    print("   2. outputs/exploration/embeddings_2d_unlabeled.png")
    print("   3. outputs/exploration/embeddings_temporal.png")
    
    print("\n🎯 Next step: Run 'python find_optimal_k.py'")


if __name__ == "__main__":
    main()