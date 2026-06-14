# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using Python, Pandas, Scikit-Learn, and Streamlit. This application recommends movies similar to the one selected by the user based on movie metadata such as genres, keywords, cast, crew, and overview.

## 🚀 Features

* Recommend 5 similar movies instantly
* Content-based filtering using cosine similarity
* Interactive and user-friendly Streamlit interface
* Optimized similarity matrix using Joblib compression
* Fast and lightweight deployment

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

## 📂 Dataset

This project uses the TMDB 5000 Movies Dataset, which contains information about thousands of movies, including:

* Movie titles
* Genres
* Cast
* Crew
* Keywords
* Overviews

## ⚙️ How It Works

1. Movie metadata is preprocessed and combined into a single feature set.
2. Text data is transformed using vectorization techniques.
3. Cosine similarity is calculated between all movies.
4. The similarity matrix is compressed and stored using Joblib.
5. When a user selects a movie, the system finds the most similar movies and displays recommendations.

## 📦 Installation

Clone the repository:

git clone https://github.com/akshatprogrammer-byte/Movie_Recom.git
cd Movie_Recommendation

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## 📸 Preview

Select a movie from the dropdown menu and click the **Recommend** button to receive similar movie suggestions.

## 👨‍💻 Author

**Akshat Mehrotra**

B.Tech CSE Student | AI/ML Enthusiast

live link: https://mov-rec-sys.streamlit.app/

⭐ If you found this project useful, consider giving it a star on GitHub.
