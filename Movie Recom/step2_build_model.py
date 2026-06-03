"""
STEP 2 — BUILD MODEL (STORAGE-SAFE)
Saves only top-50 similar movies per film.
~7400 movies → files are ~15 MB total (not GBs!)

Usage: python scripts/step2_build_model.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('model', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading processed data...")
df = pd.read_csv('data/processed.csv')
print(f"   {len(df):,} movies loaded")

# ── TF-IDF ────────────────────────────────────────────────────────────────────
# Converts each movie's soup into a vector.
# ngram_range=(1,2) captures both single words and pairs like "vicky_kaushal action"
print("\n🔢 Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(df['soup'].fillna(''))
print(f"   Matrix: {tfidf_matrix.shape[0]:,} movies × {tfidf_matrix.shape[1]:,} features")

# ── STORAGE-SAFE: Save only top-K similar movies ──────────────────────────────
# Instead of storing ALL similarities (7400×7400 = 54M values = ~200MB),
# we keep only top 50 per movie → 7400×50 = 370K values = ~3MB
TOP_K = 50
print(f"\n📐 Computing top-{TOP_K} similarities per movie...")
print(f"   Processing in batches to avoid RAM spikes...")

n = len(df)
top_indices = np.zeros((n, TOP_K), dtype=np.int32)
top_scores  = np.zeros((n, TOP_K), dtype=np.float32)

BATCH = 500   # process 500 movies at a time
for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    # similarity of this batch vs ALL movies
    batch_sim = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)

    for i, sim_row in enumerate(batch_sim):
        global_idx = start + i
        sim_row[global_idx] = -1              # exclude self
        best = np.argpartition(sim_row, -TOP_K)[-TOP_K:]   # fast top-K
        best = best[np.argsort(sim_row[best])[::-1]]       # sort descending
        top_indices[global_idx] = best
        top_scores[global_idx]  = sim_row[best]

    if end % 1000 == 0 or end == n:
        print(f"   ✅ {end:,}/{n:,} done")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\n💾 Saving model files...")
np.save('model/top_indices.npy', top_indices)
np.save('model/top_scores.npy',  top_scores)
print(f"   ✅ model/top_indices.npy  ({top_indices.nbytes / 1024:.0f} KB)")
print(f"   ✅ model/top_scores.npy   ({top_scores.nbytes  / 1024:.0f} KB)")

# Save title → index lookup
title_to_idx = pd.Series(df.index, index=df['title'].str.lower().str.strip())
title_to_idx.to_pickle('model/title_to_idx.pkl')
print(f"   ✅ model/title_to_idx.pkl")

# Save clean movie info
save_cols = ['title', 'genres_clean', 'actors_clean', 'imdb_rating',
             'imdb_votes', 'year_of_release', 'runtime', 'story',
             'summary', 'poster_path', 'wiki_link']
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv('model/movies.csv', index=False)
print(f"   ✅ model/movies.csv")

total_kb = (top_indices.nbytes + top_scores.nbytes) / 1024
print(f"\n🎉 Done! Total model size: ~{total_kb:.0f} KB")
print("Now run: python scripts/step3_recommend.py")
