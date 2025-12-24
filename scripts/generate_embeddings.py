"""
Generate embeddings for financial news headlines
Uses FinBERT for finance-domain understanding
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_embeddings(model_name='ProsusAI/finbert'):
    """
    Generate embeddings for all news headlines
    
    Args:
        model_name: Either 'ProsusAI/finbert' (768-dim, finance-specific)
                    or 'sentence-transformers/all-MiniLM-L6-v2' (384-dim, faster)
    """
    
    print("=" * 60)
    print("🔄 GENERATING EMBEDDINGS")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Load news data
    print("\n📰 Loading news data...")
    news = pd.read_csv("data/processed/final_news.csv")
    print(f"   Loaded {len(news)} articles")
    
    # Load embedding model
    print(f"\n🤖 Loading model: {model_name}")
    print("   (This may take 1-2 minutes on first run)")
    model = SentenceTransformer(model_name)
    print(f"   ✓ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    
    # Generate embeddings
    print("\n🧮 Generating embeddings...")
    print("   (This will take 2-5 minutes for 1,553 articles)")
    
    headlines = news['headline'].tolist()
    
    # Generate with progress bar
    embeddings = model.encode(
        headlines,
        show_progress_bar=True,
        batch_size=32,  # Process 32 at a time
        convert_to_numpy=True
    )
    
    # Save embeddings
    output_path = f"data/embeddings/embeddings_{model_name.split('/')[-1]}.npy"
    np.save(output_path, embeddings)
    
    # Save metadata
    metadata = {
        'model': model_name,
        'num_articles': len(news),
        'embedding_dim': embeddings.shape[1],
        'date_generated': datetime.now().isoformat(),
        'date_range': f"{news['timestamp'].min()} to {news['timestamp'].max()}"
    }
    
    metadata_path = f"data/embeddings/metadata_{model_name.split('/')[-1]}.txt"
    with open(metadata_path, 'w') as f:
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Shape: {embeddings.shape}")
    print(f"   - {embeddings.shape[0]} articles")
    print(f"   - {embeddings.shape[1]} dimensions per article")
    print(f"\nSaved to: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("\n💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Quick sanity check
    print("\n🔍 Sanity Check:")
    print(f"   Mean embedding value: {embeddings.mean():.4f}")
    print(f"   Std embedding value: {embeddings.std():.4f}")
    print(f"   Min value: {embeddings.min():.4f}")
    print(f"   Max value: {embeddings.max():.4f}")
    
    return embeddings, metadata


def test_embedding_similarity():
    """
    Quick test to verify embeddings capture semantic similarity
    """
    print("\n" + "=" * 60)
    print("🧪 TESTING SEMANTIC SIMILARITY")
    print("=" * 60)
    
    news = pd.read_csv("data/processed/final_news.csv")
    embeddings = np.load("data/embeddings/embeddings_finbert.npy")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Pick a random article
    test_idx = 100
    test_headline = news.iloc[test_idx]['headline']
    test_embedding = embeddings[test_idx].reshape(1, -1)
    
    # Find most similar articles
    similarities = cosine_similarity(test_embedding, embeddings)[0]
    top_5_indices = similarities.argsort()[-6:-1][::-1]  # Top 5 excluding itself
    
    print(f"\n📄 Test headline:\n   '{test_headline}'")
    print(f"\n🔍 Top 5 most similar headlines:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"\n   {i}. Similarity: {similarities[idx]:.3f}")
        print(f"      '{news.iloc[idx]['headline']}'")
    
    print("\n✅ If similar headlines are semantically related, embeddings are working!")


if __name__ == "__main__":
    # Generate embeddings with FinBERT (finance-specific)
    embeddings, metadata = generate_embeddings(model_name='ProsusAI/finbert')
    
    # Optional: Test similarity
    print("\n" + "=" * 60)
    response = input("Run similarity test? (y/n): ")
    if response.lower() == 'y':
        test_embedding_similarity()
    
    print("\n🎯 Next step: Run 'python explore_embeddings.py'")