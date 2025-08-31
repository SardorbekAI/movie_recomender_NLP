import difflib
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    return movies


# Preprocess features
@st.cache_data
def preprocess_data(movies):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies[feature] = movies[feature].fillna('')

    combined_features = (
        movies['genres'] + ' ' +
        movies['keywords'] + ' ' +
        movies['tagline'] + ' ' +
        movies['cast'] + ' ' +
        movies['director']
    )

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity_matrix = cosine_similarity(feature_vectors)

    return similarity_matrix


# Recommend movies
def recommend_movies(movie_name, movies, similarity_matrix):
    list_of_all_titles = movies['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        return []

    close_match = close_matches[0]
    index_of_the_movie = movies[movies.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity_matrix[index_of_the_movie]))
    sorted_similar_movies = sorted(
        similarity_score, key=lambda x: x[1], reverse=True
    )

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:11], start=1):
        index = movie[0]
        title_from_index = movies[movies.index == index]['title'].values[0]
        recommended_movies.append((i, title_from_index))

    return recommended_movies


# Streamlit UI
def main():
    st.set_page_config(
        page_title="🎬 Movie Recommender",
        page_icon="🍿",
        layout="centered"
    )

    st.title("🎥 Movie Recommendation System")
    st.markdown("Find your next favorite movie based on what you love! 🍿✨")

    st.divider()

    movies = load_data()
    similarity_matrix = preprocess_data(movies)

    movie_name = st.text_input("🔍 Enter your favorite movie:")

    if st.button("✨ Recommend Movies"):
        if movie_name.strip() == "":
            st.warning("⚠️ Please enter a movie name.")
        else:
            recommendations = recommend_movies(
                movie_name, movies, similarity_matrix
            )
            if recommendations:
                st.success("✅ Here are the top 10 movies for you:")
                for idx, title in recommendations:
                    st.write(f"{idx}. 🎬 **{title}**")
            else:
                st.error("❌ Sorry, no similar movies found. Try another one!")


if __name__ == "__main__":
    main()
