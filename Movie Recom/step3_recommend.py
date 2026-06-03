"""
STEP 3 — BOLLYWOOD RECOMMENDATION ENGINE
Four ways to find movies:
  1. recommend(title)         → similar movies by plot + cast + genre
  2. hybrid(title)            → similar + high rated
  3. by_actor(name)           → movies featuring an actor
  4. by_genre(genre)          → top rated movies in a genre

Usage: python scripts/step3_recommend.py
"""

import pandas as pd
import numpy as np

# ── Load model ────────────────────────────────────────────────────────────────
print("📂 Loading model...")
try:
    top_indices  = np.load('model/top_indices.npy')
    top_scores   = np.load('model/top_scores.npy')
    title_to_idx = pd.read_pickle('model/title_to_idx.pkl')
    df           = pd.read_csv('model/movies.csv')
    print(f"   ✅ {len(df):,} Bollywood movies ready\n")
except FileNotFoundError:
    print("❌ Model not found. Run step2_build_model.py first!")
    exit(1)


# ── Lookup helper with fuzzy fallback ─────────────────────────────────────────
def find_movie(title):
    """Returns (index, matched_title) or (None, None)."""
    key = title.strip().lower()

    if key in title_to_idx:
        return int(title_to_idx[key]), df.iloc[int(title_to_idx[key])]['title']

    # Partial match fallback
    matches = [(t, title_to_idx[t]) for t in title_to_idx.index if key in t]
    if matches:
        print(f"   '{title}' not found exactly. Closest matches:")
        for m, _ in matches[:5]:
            print(f"     → {df.iloc[title_to_idx[m]]['title']}")
        # Auto-use first match
        best = matches[0]
        print(f"   Using: '{df.iloc[best[1]]['title']}'\n")
        return int(best[1]), df.iloc[best[1]]['title']

    print(f"❌ '{title}' not found. Try another title.")
    return None, None


# ── Print helper ──────────────────────────────────────────────────────────────
def _show(results, score_col='score', title_header=''):
    if title_header:
        print(title_header)
    print()
    for _, row in results.iterrows():
        rating = f"⭐ {row['imdb_rating']:.1f}" if pd.notna(row.get('imdb_rating')) else ''
        year   = f"({int(row['year_of_release'])})" if pd.notna(row.get('year_of_release')) else ''
        score  = f"[{row.get(score_col, 0):.3f}]" if score_col in row else ''
        genres = row.get('genres_clean', '').replace('_', ' ').title()[:30]
        actors = row.get('actors_clean', '').replace('_', ' ').replace('  ', ', ')[:40]
        print(f"  🎬 {row['title']} {year} {rating} {score}")
        print(f"     Genre: {genres}")
        print(f"     Cast:  {actors}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTENT-BASED: similar plot + cast + genre
# ─────────────────────────────────────────────────────────────────────────────
def recommend(movie_title, n=8):
    """Find movies similar in story, cast, and genre."""
    idx, matched = find_movie(movie_title)
    if idx is None:
        return

    indices = top_indices[idx][:n]
    scores  = top_scores[idx][:n]

    results = df.iloc[indices].copy()
    results['score'] = scores.round(4)

    _show(results, 'score',
          f"🎬 Because you liked '{matched}' — Content similar movies:")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. HYBRID: similar + quality boost via Bayesian rating
# ─────────────────────────────────────────────────────────────────────────────
def hybrid(movie_title, n=8, alpha=0.60):
    """
    Blend of content similarity and IMDB quality.
    alpha=0.60 → 60% similarity, 40% rating quality
    Lower alpha = more focus on highly-rated films.
    """
    idx, matched = find_movie(movie_title)
    if idx is None:
        return

    # Get top-50 candidates
    cand_idx    = top_indices[idx]
    cand_scores = top_scores[idx]

    candidates = df.iloc[cand_idx].copy()
    candidates['sim_score'] = cand_scores

    # Bayesian weighted rating (IMDB formula)
    # Prevents a movie with 10 votes / 9.5 rating beating one with 10000 votes / 8.0
    C = df['imdb_rating'].mean()
    m = df['imdb_votes'].quantile(0.40)   # 40th percentile as threshold
    v = candidates['imdb_votes'].fillna(0)
    R = candidates['imdb_rating'].fillna(C)
    candidates['quality'] = (v / (v + m)) * R + (m / (v + m)) * C

    # Normalise both to [0,1]
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s * 0

    candidates['final'] = (
        alpha * norm(candidates['sim_score']) +
        (1 - alpha) * norm(candidates['quality'])
    ).round(4)

    results = candidates.nlargest(n, 'final')
    _show(results, 'final',
          f"🏆 Hybrid picks for '{matched}' (similarity + quality):")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. BY ACTOR: all movies featuring an actor
# ─────────────────────────────────────────────────────────────────────────────
def by_actor(actor_name, n=10):
    """
    Find top-rated movies featuring a specific actor.
    Works with partial names: 'shahrukh', 'aamir', 'deepika' etc.
    """
    key = actor_name.strip().lower().replace(' ', '_')
    mask = df['actors_clean'].fillna('').str.lower().str.contains(key)
    results = df[mask].copy()

    if results.empty:
        # Try without underscore
        key2 = actor_name.strip().lower()
        mask2 = df['actors_clean'].fillna('').str.lower().str.contains(key2)
        results = df[mask2].copy()

    if results.empty:
        print(f"❌ No movies found with actor '{actor_name}'")
        return None

    results = results.sort_values('imdb_rating', ascending=False).head(n)
    _show(results, title_header=f"🎭 Top movies with '{actor_name}':")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. BY GENRE: best rated movies in a genre
# ─────────────────────────────────────────────────────────────────────────────
def by_genre(genre, n=10, min_votes=200):
    """
    Best Bollywood movies in a genre.
    Examples: 'action', 'romance', 'drama', 'comedy', 'thriller', 'war'
    """
    key = genre.strip().lower().replace(' ', '_')
    mask = df['genres_clean'].fillna('').str.lower().str.contains(key)
    results = df[mask & (df['imdb_votes'].fillna(0) >= min_votes)].copy()

    if results.empty:
        print(f"❌ No movies found for genre '{genre}'")
        return None

    # Bayesian rating for fair ranking
    C = df['imdb_rating'].mean()
    v = results['imdb_votes'].fillna(0)
    R = results['imdb_rating'].fillna(C)
    results['score'] = ((v / (v + min_votes)) * R + (min_votes / (v + min_votes)) * C).round(3)

    results = results.nlargest(n, 'score')
    _show(results, 'score', f"🎭 Best '{genre}' Bollywood movies:")
    return results


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  🎬 BOLLYWOOD RECOMMENDER — Demo")
    print("=" * 60 + "\n")

    recommend("Uri: The Surgical Strike")
    hybrid("Dangal")
    by_actor("Shahrukh Khan")
    by_genre("romance")

    # ── Interactive ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  INTERACTIVE MODE  (type 'q' to quit)")
    print("=" * 60)

    while True:
        print("\n[1] Similar movies  [2] Hybrid  [3] By actor  [4] By genre  [q] Quit")
        choice = input("Choice: ").strip()

        if choice == 'q':
            break
        elif choice == '1':
            recommend(input("Movie title: "), int(input("How many? [8]: ") or 8))
        elif choice == '2':
            hybrid(input("Movie title: "), int(input("How many? [8]: ") or 8))
        elif choice == '3':
            by_actor(input("Actor name (e.g. Aamir Khan): "))
        elif choice == '4':
            by_genre(input("Genre (e.g. action, romance, thriller): "))
        else:
            print("Invalid. Choose 1–4 or q.")
