import streamlit as st
import pickle
import pandas as pd
import requests
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]]['id']
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

movie_dict = pickle.load(open('movies.pkl' , 'rb'))
movies = pd.DataFrame(movie_dict)
similarity = joblib.load('similarity.joblib')
st.title('Movie Recommendation System')
select_movie_name = st.selectbox('Select a movie',movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(select_movie_name)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("*", movie)
st.caption("Made with ❤️ by Akshat Mehrotra")
