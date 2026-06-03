# 🎬 Bollywood Movie Recommender

Content-based + hybrid recommender for ~7,400 Bollywood movies.
Model files stay under **30 MB total** — no storage issues!

## Setup

```bash
pip install pandas scikit-learn numpy
```

Put your CSV as `databollywood.csv`, then run in order:

```bash
python scripts/step1_preprocess.py    # ~10 seconds
python scripts/step2_build_model.py   # ~1-2 minutes
python scripts/step3_recommend.py     # instant!
```

## Four Ways to Recommend

```python
from scripts.step3_recommend import recommend, hybrid, by_actor, by_genre

recommend("Dangal")                  # similar story/cast/genre
hybrid("3 Idiots", alpha=0.6)        # similar + high rated
by_actor("Aamir Khan")               # all Aamir Khan movies ranked
by_genre("thriller")                 # best thrillers
```

## Why So Fast & Small?

Instead of saving a full 7400×7400 similarity matrix (~200 MB),
we save only the **top 50 most similar movies per film**:

```
Full matrix:  7400 × 7400 × 4 bytes = ~220 MB
Top-50 only:  7400 ×   50 × 4 bytes =  ~1.5 MB ✅
```

## How Similarity Works

Each movie becomes a "soup" text:
```
actors (×3 weight) + genres (×3) + story (×2) + summary + tagline
```

Repetition = higher weight. So actor names strongly influence matching —
searching "Uri" will find other Vicky Kaushal or war movies.

TF-IDF converts this to a vector → cosine similarity finds the angle
between two movies' vectors → smaller angle = more similar.
