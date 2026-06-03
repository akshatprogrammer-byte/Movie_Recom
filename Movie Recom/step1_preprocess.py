"""
STEP 1 — PREPROCESS BOLLYWOOD DATASET
Cleans data and builds a 'soup' text for each movie.
Usage: python scripts/step1_preprocess.py
"""

import pandas as pd
import re
import os

os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv('databollywood.csv')
print(f"   Loaded: {len(df):,} rows × {df.shape[1]} columns")

# ── 1. Use best available title ───────────────────────────────────────────────
# title_x and title_y are both title columns — use title_x, fallback to title_y
df['title'] = df['title_x'].fillna(df['title_y']).fillna(df['original_title'])
df = df.dropna(subset=['title'])
df['title'] = df['title'].str.strip()

# ── 2. Drop duplicates ────────────────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset=['title'], keep='first')
print(f"   Removed {before - len(df)} duplicate titles")

# ── 3. Clean text helper ──────────────────────────────────────────────────────
def clean(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── 4. Parse pipe-separated actors  e.g. "Vicky Kaushal|Paresh Rawal|..." ────
def parse_actors(text, max_actors=5):
    """Take first 5 actors, clean names, replace spaces with underscore."""
    if pd.isna(text):
        return ''
    actors = str(text).split('|')[:max_actors]
    cleaned = []
    for a in actors:
        a = a.strip()
        if a:
            # "Vicky Kaushal" → "vicky_kaushal" (treated as one token by TF-IDF)
            cleaned.append(re.sub(r'\s+', '_', a.lower()))
    return ' '.join(cleaned)

# ── 5. Parse genres  e.g. "Action|Drama|War" ─────────────────────────────────
def parse_genres(text):
    if pd.isna(text):
        return ''
    genres = str(text).split('|')
    return ' '.join(g.strip().replace(' ', '_').lower() for g in genres if g.strip())

# ── 6. Apply parsing ──────────────────────────────────────────────────────────
print("🔧 Parsing columns...")
df['genres_clean']  = df['genres'].apply(parse_genres)
df['actors_clean']  = df['actors'].apply(parse_actors)
df['story_clean']   = df['story'].apply(clean)
df['summary_clean'] = df['summary'].apply(clean)
df['tagline_clean'] = df['tagline'].apply(clean)

# ── 7. Build SOUP ─────────────────────────────────────────────────────────────
# Weights via repetition:
#   actors × 3  (people search "Shahrukh Khan movies")
#   genres × 3  (people search "romantic Bollywood movies")
#   story  × 2  (plot is important)
#   summary × 1
#   tagline × 1
df['soup'] = (
    df['actors_clean']  + ' ' + df['actors_clean']  + ' ' + df['actors_clean']  + ' ' +
    df['genres_clean']  + ' ' + df['genres_clean']  + ' ' + df['genres_clean']  + ' ' +
    df['story_clean']   + ' ' + df['story_clean']   + ' ' +
    df['summary_clean'] + ' ' +
    df['tagline_clean']
).str.strip()

# ── 8. Filter quality ─────────────────────────────────────────────────────────
before = len(df)
df['imdb_votes'] = pd.to_numeric(df['imdb_votes'], errors='coerce').fillna(0)
df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(0)
df = df[df['soup'].str.len() > 20]     # must have some content
print(f"   Removed {before - len(df)} rows with empty content")
print(f"   ✅ Final: {len(df):,} movies")

# ── 9. Save ───────────────────────────────────────────────────────────────────
df = df.reset_index(drop=True)
df.to_csv('data/processed.csv', index=False)

print("\n📊 Sample processed data:")
print(df[['title', 'genres_clean', 'actors_clean']].head(5).to_string())
print(f"\n✅ Saved → data/processed.csv")
print("Now run: python scripts/step2_build_model.py")
