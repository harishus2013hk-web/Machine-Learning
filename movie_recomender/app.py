import streamlit as st
import pickle
import pandas as pd
import os
import gdown

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #FAFAFA;
        margin-bottom: 30px;
    }
    .movie-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 12px;
        color: white;
        font-size: 18px;
        font-weight: 500;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    .stSelectbox label {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Download similarity.pkl from Google Drive (only on first run)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading recommendation model, please wait...")
def load_models():
    if not os.path.exists("similarity.pkl"):
        file_id = "1SWFce8D8CbhAqP1Qlwreb3sTorn5eOCK"        # <── paste your Google Drive file ID here
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "similarity.pkl", quiet=False)

    movies = pickle.load(open("movies.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

movies_list, similarity = load_models()

# ─────────────────────────────────────────────────────────────────
# Recommendation Function
# ─────────────────────────────────────────────────────────────────
def recommend(movie):
    idx = movies_list[movies_list['title'] == movie].index[0]
    distance = similarity[idx]
    movies = sorted(
        list(enumerate(distance)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    recommended_movies = []
    for i in movies:
        recommended_movies.append(movies_list.iloc[i[0]].title)
    return recommended_movies

# ─────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Find movies similar to your favorite one</div>',
    unsafe_allow_html=True
)

selected_movie = st.selectbox(
    "Choose a Movie",
    movies_list['title'].values
)

if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)
    st.markdown("## Recommended Movies 🍿")
    for movie in recommendations:
        st.markdown(
            f'<div class="movie-card">⭐ {movie}</div>',
            unsafe_allow_html=True
        )
