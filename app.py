from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"

st.set_page_config(page_title="Film Recommendation", page_icon="🎬", layout="wide")
st.title("Film Recommendation")
st.caption("Interactive Streamlit version of the film recommender project.")


@st.cache_data
def load_data():
    if not MOVIES_FILE.exists() or not RATINGS_FILE.exists():
        return None, None
    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    return movies, ratings


@st.cache_data
def compute_similarity_matrix(genres_series: pd.Series):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(genres_series)
    return cosine_similarity(tfidf_matrix)


@st.cache_data
def compute_rating_stats(ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
    return (
        ratings_df.merge(movies_df, on="movieId", how="left")
        .groupby("title")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
    )


movies, ratings = load_data()

if movies is None or ratings is None:
    st.warning("Place `movies.csv` and `ratings.csv` inside `data/`.")
    st.stop()

# Sidebar Form untuk Filter
genre_list = sorted({g for s in movies["genres"].dropna() for g in s.split("|")})

with st.sidebar.form("filter_form"):
    st.header("Filters")
    genre_filter = st.multiselect("Genres", genre_list, default=genre_list[:8])
    min_ratings = st.slider("Min ratings per movie", 1, 500, 50)
    submit_filters = st.form_submit_button("Apply Filters")

movie_genres = movies.assign(genres_split=movies["genres"].str.split("|")).explode("genres_split")

# Multi-select fallback jika kosong
if not genre_filter:
    genre_filter = genre_list

filtered_movies = movies[
    movies["genres"].fillna("").apply(lambda x: any(g in x.split("|") for g in genre_filter))
].copy()

col1, col2 = st.columns(2)
col1.metric("Total Movies", f"{len(movies):,}".replace(",", "."))
col2.metric("Total Ratings", f"{len(ratings):,}".replace(",", "."))

tab_overview, tab_reco, tab_data = st.tabs(["Overview", "Recommend", "Data"])

# Pre-compute similarity matriks satu kali via cache
sim_matrix = compute_similarity_matrix(movies["genres_clean"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        genre_counts = (
            movie_genres[movie_genres["genres_split"].isin(genre_filter)]["genres_split"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        genre_counts.columns = ["genre", "count"]
        fig_genre = px.bar(genre_counts, x="count", y="genre", orientation="h", title="Top Filtered Genres")
        fig_genre.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_genre, use_container_width=True)
    with right:
        rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig_rating = px.bar(rating_counts, x="rating", y="count", title="Rating Distribution")
        st.plotly_chart(fig_rating, use_container_width=True)

with tab_reco:
    st.subheader("🎬 Movie Recommender")

    with st.form("recommendation_form"):
        title = st.selectbox("Pick an anchor movie", movies["title"].sort_values().tolist())
        top_n = st.slider("Top N recommendations", 3, 10, 5)
        submit_reco = st.form_submit_button("Get Recommendations")

    # Ambil matriks similarity dari cache secara instant
    idx = movies.index[movies["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    rec = movies.iloc[[i for i, _ in scores]][["title", "genres"]].copy()
    rec["similarity"] = [round(s, 3) for _, s in scores]

    st.markdown("### Recommendation Results")
    st.dataframe(rec, use_container_width=True)
    st.caption("Similarity is calculated using genre TF-IDF + cosine similarity (cached for instant performance).")

with tab_data:
    st.subheader("Filtered Movie Statistics")
    movie_rating_stats = compute_rating_stats(ratings, movies)
    movie_rating_stats = movie_rating_stats[movie_rating_stats["rating_count"] >= min_ratings]
    movie_rating_stats = movie_rating_stats.sort_values(
        ["avg_rating", "rating_count"], ascending=[False, False]
    ).head(250)
    st.dataframe(movie_rating_stats, use_container_width=True)
