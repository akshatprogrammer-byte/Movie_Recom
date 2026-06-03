"""
🎬 BOLLYWOOD MOVIE RECOMMENDER — STREAMLIT APP
Beautiful web UI for movie recommendations.

Run with: streamlit run app.py
Then open: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from functools import lru_cache

st.set_page_config(
    page_title="🎬 Bollywood Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: #f8f9fa;
    }
    .movie-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .rating {
        color: #ffc107;
        font-size: 18px;
        font-weight: bold;
    }
    .title {
        color: #333;
        font-size: 16px;
        font-weight: bold;
        margin: 10px 0;
    }
    .meta {
        color: #666;
        font-size: 12px;
    }
    .genre-tag {
        display: inline-block;
        background: #e9ecef;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 4px 4px 4px 0;
        font-size: 11px;
        color: #495057;
    }
    .actor-name {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 4px 4px 4px 0;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model files once."""
    try:
        top_indices  = np.load('model/top_indices.npy')
        top_scores   = np.load('model/top_scores.npy')
        title_to_idx = pd.read_pickle('model/title_to_idx.pkl')
        df           = pd.read_csv('model/movies.csv')
        return top_indices, top_scores, title_to_idx, df
    except FileNotFoundError:
        st.error("❌ Model not found! Run the Python scripts first:")
        st.code("""
python scripts/step1_preprocess.py
python scripts/step2_build_model.py
        """)
        st.stop()

top_indices, top_scores, title_to_idx, df = load_model()

# ── RECOMMENDATION FUNCTIONS ──────────────────────────────────────────────────
def find_movie(title):
    """Find movie by title with fuzzy matching."""
    key = title.strip().lower()
    if key in title_to_idx:
        return int(title_to_idx[key]), df.iloc[int(title_to_idx[key])]['title']

    matches = [(t, title_to_idx[t]) for t in title_to_idx.index if key in t]
    if matches:
        idx = int(matches[0][1])
        return idx, df.iloc[idx]['title']
    return None, None

def recommend_similar(movie_title, n=8):
    """Content-based recommendations."""
    idx, matched = find_movie(movie_title)
    if idx is None:
        return None, None

    indices = top_indices[idx][:n]
    scores  = top_scores[idx][:n]
    results = df.iloc[indices].copy()
    results['score'] = scores
    return matched, results.to_dict('records')

def recommend_hybrid(movie_title, n=8, alpha=0.60):
    """Hybrid: similar + quality."""
    idx, matched = find_movie(movie_title)
    if idx is None:
        return None, None

    cand_idx    = top_indices[idx]
    cand_scores = top_scores[idx]
    candidates = df.iloc[cand_idx].copy()
    candidates['sim_score'] = cand_scores

    C = df['imdb_rating'].mean()
    m = df['imdb_votes'].quantile(0.40)
    v = candidates['imdb_votes'].fillna(0)
    R = candidates['imdb_rating'].fillna(C)
    candidates['quality'] = (v / (v + m)) * R + (m / (v + m)) * C

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s * 0

    candidates['final'] = (
        alpha * norm(candidates['sim_score']) +
        (1 - alpha) * norm(candidates['quality'])
    )

    results = candidates.nlargest(n, 'final')
    return matched, results.to_dict('records')

def get_by_actor(actor_name, n=10):
    """Movies by actor."""
    key = actor_name.strip().lower().replace(' ', '_')
    mask = df['actors_clean'].fillna('').str.lower().str.contains(key)
    results = df[mask]

    if results.empty:
        key2 = actor_name.strip().lower()
        mask2 = df['actors_clean'].fillna('').str.lower().str.contains(key2)
        results = df[mask2]

    if results.empty:
        return None

    results = results.sort_values('imdb_rating', ascending=False).head(n)
    return results.to_dict('records')

def get_by_genre(genre, n=10, min_votes=200):
    """Movies by genre."""
    key = genre.strip().lower().replace(' ', '_')
    mask = df['genres_clean'].fillna('').str.lower().str.contains(key)
    results = df[mask & (df['imdb_votes'].fillna(0) >= min_votes)]

    if results.empty:
        return None

    C = df['imdb_rating'].mean()
    v = results['imdb_votes'].fillna(0)
    R = results['imdb_rating'].fillna(C)
    results['score'] = (v / (v + min_votes)) * R + (min_votes / (v + min_votes)) * C

    results = results.nlargest(n, 'score')
    return results.to_dict('records')

# ── DISPLAY MOVIE CARD ────────────────────────────────────────────────────────
def movie_card(movie, col):
    """Display a single movie card."""
    with col:
        st.markdown(f"""
        <div class="movie-card">
        """, unsafe_allow_html=True)

        # Poster with emoji fallback (no external image loading)
        poster_html = f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            padding: 60px 20px;
            text-align: center;
            color: white;
            font-size: 48px;
            margin-bottom: 15px;
        '>
            🎬
        </div>
        """
        st.markdown(poster_html, unsafe_allow_html=True)

        # Title
        st.markdown(f"<div class='title'>{movie['title']}</div>", unsafe_allow_html=True)

        # Rating & Year
        col1, col2 = st.columns(2)
        with col1:
            if pd.notna(movie.get('imdb_rating')):
                st.markdown(f"<div class='rating'>⭐ {movie['imdb_rating']:.1f}</div>", unsafe_allow_html=True)
        with col2:
            if pd.notna(movie.get('year_of_release')):
                st.markdown(f"<div class='meta'>📅 {int(movie['year_of_release'])}</div>", unsafe_allow_html=True)

        # Genres
        if pd.notna(movie.get('genres_clean')):
            genres = movie['genres_clean'].replace('_', ' ').split()
            genre_html = ''.join([f"<span class='genre-tag'>{g}</span>" for g in genres[:3]])
            st.markdown(f"<div>{genre_html}</div>", unsafe_allow_html=True)

        # Cast
        if pd.notna(movie.get('actors_clean')):
            actors = movie['actors_clean'].replace('_', ' ').split()[:3]
            actor_html = ''.join([f"<span class='actor-name'>{a}</span>" for a in actors])
            st.markdown(f"<div>{actor_html}</div>", unsafe_allow_html=True)

        # Runtime
        if pd.notna(movie.get('runtime')):
            st.markdown(f"<div class='meta'>⏱️ {int(movie['runtime'])} min</div>", unsafe_allow_html=True)

        # Story snippet
        if pd.notna(movie.get('story')):
            story = str(movie['story'])[:100]
            st.caption(f"📖 {story}...")

        st.markdown("</div>", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align: center; color: #667eea;'>🎬 Bollywood Movie Recommender</h1>
<p style='text-align: center; color: #666;'>Find your next favorite Bollywood movie</p>
""", unsafe_allow_html=True)

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Similar Movies", "🏆 Hybrid", "🎭 By Actor", "🎪 By Genre"])

# ── TAB 1: Similar Movies ─────────────────────────────────────────────────────
with tab1:
    st.header("🎯 Find Similar Movies")
    st.write("Discover movies based on plot, cast, and genre")

    col1, col2 = st.columns([3, 1])
    with col1:
        movie_title = st.text_input(
            "Enter a Bollywood movie title",
            value="Uri: The Surgical Strike",
            key="similar_title"
        )
    with col2:
        n_recs = st.slider("How many?", 1, 20, 8, key="similar_n")

    if movie_title:
        if st.button("🔍 Find Similar", key="btn_similar"):
            with st.spinner("Finding similar movies..."):
                matched, movies = recommend_similar(movie_title, n_recs)

                if matched is None:
                    st.error(f"❌ '{movie_title}' not found. Try another title.")
                else:
                    st.success(f"✅ Because you liked **{matched}**")
                    cols = st.columns(4)
                    for i, movie in enumerate(movies):
                        movie_card(movie, cols[i % 4])

# ── TAB 2: Hybrid Recommendations ─────────────────────────────────────────────
with tab2:
    st.header("🏆 Hybrid Recommendations")
    st.write("Similar movies + high quality (IMDB ratings)")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        movie_title_hybrid = st.text_input(
            "Enter movie title",
            value="Dangal",
            key="hybrid_title"
        )
    with col2:
        n_hybrid = st.slider("How many?", 1, 20, 8, key="hybrid_n")
    with col3:
        alpha = st.slider("Similarity weight", 0.0, 1.0, 0.60, 0.05, key="alpha")

    if movie_title_hybrid:
        if st.button("🔍 Get Hybrid Recs", key="btn_hybrid"):
            with st.spinner("Computing hybrid score..."):
                matched, movies = recommend_hybrid(movie_title_hybrid, n_hybrid, alpha)

                if matched is None:
                    st.error(f"❌ '{movie_title_hybrid}' not found.")
                else:
                    st.success(f"✅ Best for **{matched}** ({int(alpha*100)}% similarity, {int((1-alpha)*100)}% quality)")
                    cols = st.columns(4)
                    for i, movie in enumerate(movies):
                        movie_card(movie, cols[i % 4])

# ── TAB 3: By Actor ───────────────────────────────────────────────────────────
with tab3:
    st.header("🎭 Movies by Actor")
    st.write("Top-rated films featuring a specific actor")

    col1, col2 = st.columns([3, 1])
    with col1:
        actor_name = st.text_input(
            "Actor name (e.g. Shahrukh Khan, Aamir Khan, Deepika Padukone)",
            value="Aamir Khan",
            key="actor_input"
        )
    with col2:
        n_actor = st.slider("How many?", 1, 20, 10, key="actor_n")

    if actor_name:
        if st.button("🔍 Find Movies", key="btn_actor"):
            with st.spinner(f"Finding {actor_name} movies..."):
                movies = get_by_actor(actor_name, n_actor)

                if movies is None:
                    st.error(f"❌ No movies found for '{actor_name}'")
                else:
                    st.success(f"✅ Top {len(movies)} movies with **{actor_name}**")
                    cols = st.columns(4)
                    for i, movie in enumerate(movies):
                        movie_card(movie, cols[i % 4])

# ── TAB 4: By Genre ──────────────────────────────────────────────────────────
with tab4:
    st.header("🎪 Best Movies by Genre")
    st.write("Top-rated films in your favorite genre")

    genres = sorted([g for genres_str in df['genres_clean'].dropna()
                     for g in genres_str.split() if g])
    genres = list(set(genres))

    col1, col2 = st.columns([2, 1])
    with col1:
        genre = st.selectbox(
            "Choose a genre",
            options=genres,
            index=0 if 'action' in genres else 0,
            key="genre_select"
        )
    with col2:
        n_genre = st.slider("How many?", 1, 20, 10, key="genre_n")

    if st.button("🔍 Show Best", key="btn_genre"):
        with st.spinner(f"Finding best {genre} movies..."):
            movies = get_by_genre(genre, n_genre)

            if movies is None:
                st.error(f"❌ No movies found for genre '{genre}'")
            else:
                st.success(f"✅ Best **{genre}** Bollywood movies")
                cols = st.columns(4)
                for i, movie in enumerate(movies):
                    movie_card(movie, cols[i % 4])

# ── SIDEBAR INFO ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Dataset Info")
    st.metric("Total Movies", f"{len(df):,}")
    st.metric("Avg Rating", f"{df['imdb_rating'].mean():.2f} ⭐")
    st.metric("Avg Votes", f"{int(df['imdb_votes'].mean()):,}")

    st.divider()

    st.header("💡 How it works")
    st.write("""
    **Content-Based Filtering:**
    - Each movie is converted to text (plot + cast + genres)
    - TF-IDF finds which words are most important
    - Cosine similarity finds movies with similar "DNA"

    **Hybrid:**
    - Blends similarity with IMDB ratings
    - Prevents obscure high-rated films from dominating

    **By Actor & Genre:**
    - Direct search + Bayesian quality ranking
    """)

    st.divider()
    st.caption("Built with ❤️ using Streamlit")