import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample Movie Dataset
data = {
    "MovieID": [1, 2, 3, 4, 5],
    "Title": ["The Matrix", "Inception", "Interstellar", "The Notebook", "Titanic"],
    "Genres": ["Action Sci-Fi", "Action Sci-Fi Thriller", "Sci-Fi Drama", "Romance Drama", "Romance Drama"],
}

movies = pd.DataFrame(data)

# User preferences (example user likes "Action" and "Sci-Fi")
user_profile = {"Action": 3, "Sci-Fi": 3, "Thriller": 2, "Romance": 1, "Drama": 1}
# Create a "bag of words" for movie genres
movies["Genres_Bag"] = movies["Genres"].str.replace(" ", "")
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["Genres_Bag"])

# Calculate cosine similarity between movies based on genres
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Add user preferences
def calculate_user_score(movie, user_profile):
    movie_genres = movie.split(" ")
    return sum(user_profile.get(genre, 0) for genre in movie_genres)

# Compute scores for each movie based on user preferences
movies["UserScore"] = movies["Genres"].apply(calculate_user_score, user_profile=user_profile)

# Recommend movies with highest UserScore
recommendations = movies.sort_values(by="UserScore", ascending=False)
print(recommendations[["Title", "UserScore"]])
