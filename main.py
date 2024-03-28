import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Define retry strategy for API requests
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session = requests.Session()
session.mount('https://', adapter)

@st.cache_data  # Cache the data loading and processing for better performance
def load_data():
    movie_df = pd.read_pickle('movies.pkl')
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movie_df, similarity

def fetch_poster(movie_id):
    try:
        response = session.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=d7c9f5ec20f170114488acba1e66150c&language=en-US', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return "http://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie ID {movie_id}: {e}")
        return None

def recommend(movie, movie_df, similarity):
    movie_index = movie_df[movie_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        movie_id = i[0]
        recommended_movies.append(movie_df.iloc[movie_id].title)
        poster_url = fetch_poster(movie_id)
        if poster_url:
            recommended_movies_posters.append(poster_url)

    return recommended_movies, recommended_movies_posters

# Load data and similarity matrix
movie_df, similarity = load_data()

st.title("Movie Recommender System")

selected_movie = st.selectbox('Select a movie', movie_df['title'])

if st.button('Recommend'):
    recommendations, posters = recommend(selected_movie, movie_df, similarity)
    cols = st.columns(5)  # Use st.columns instead of st.beta_columns
    for i, col in enumerate(cols):
        with col:
            if i < len(recommendations):
                st.header(recommendations[i])
                if i < len(posters):
                    st.image(posters[i], caption=recommendations[i], use_column_width=True)
                else:
                    st.write("Poster not available")
            else:
                st.write("No recommendation available")