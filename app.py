# Import Flask framework
from flask import Flask, render_template, request

# Import library
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import linear_kernel

# Read the dataset
restaurant_review = pd.read_csv('Restaurant reviews.csv')
restaurant_metadata = pd.read_csv('Restaurant names and Metadata (1).csv')

# Merge and preprocess the data
# Filling missing values with 0 in both DataFrames
restaurant_review = restaurant_review.fillna(0)
restaurant_metadata = restaurant_metadata.fillna(0)
print(restaurant_review.shape)
print(restaurant_review.head())

# Assuming restaurant_review is your DataFrame
restaurant_review['Rating'] = pd.to_numeric(restaurant_review['Rating'], errors='coerce')

# Now create the pivot table
userRatings = restaurant_review.pivot_table(index=['Reviewer'], columns=['Restaurant'], values='Rating')
userRatings.head()
print("\nBefore: ", userRatings.shape)

# Drop restaurant_metadata with less than 10 restaurant_review and fill NaN values with 0
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0, axis=1)
print("\nAfter: ", userRatings.shape)

# Compute the correlation matrix
print()
corrMatrix = userRatings.corr(method='pearson')
print(corrMatrix.head(100))
print()

# Function to get similar restaurant_metadata
def get_similar(restaurant_name, rating):
    similar_ratings = corrMatrix[restaurant_name] * (rating - 2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

# Menggabungkan informasi dari beberapa kolom menjadi satu kolom teks
restaurant_metadata['combined_info'] = restaurant_metadata['Name'].astype(str) + ' ' + restaurant_metadata['Collections'].astype(str) + ' ' + restaurant_metadata['Cuisines'].astype(str)

# Menggunakan TF-IDF Vectorizer untuk mengonversi teks ke vektor
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(restaurant_metadata['combined_info'].fillna(''))

# Menghitung cosine similarity antara restoran
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi berbasis konten
def get_content_recommendations(restaurant_name, cosine_similarities=cosine_similarities):
    try:
        idx = restaurant_metadata[restaurant_metadata['Name'] == restaurant_name].index[0]
    except IndexError:
        return ["Restaurant not found. Please try another one."]

    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    restaurant_indices = [i[0] for i in sim_scores]

    # Membuat DataFrame hasil rekomendasi berbasis konten
    recommendations_df = restaurant_metadata.iloc[restaurant_indices]['Name']
    return recommendations_df.tolist()
# Compute the correlation matrix for user-based collaborative filtering
cosine_similarities_users = cosine_similarity(userRatings)

# Function to get collaborative filtering recommendations
def get_collaborative_recommendations(user, userRatings=userRatings, cosine_similarities_users=cosine_similarities_users):
    try:
        idx = userRatings.index.get_loc(user)
    except KeyError:
        return ["User not found. Please try another one."]

    sim_scores_user = list(enumerate(cosine_similarities_users[idx]))

    # Check if there are any recommendations
    if not sim_scores_user:
        return ["No collaborative recommendations available."]

    sim_scores_user = sorted(sim_scores_user, key=lambda x: x[1], reverse=True)
    sim_scores_user = sim_scores_user[:min(5, len(sim_scores_user))]  # Take the top 5 or all if less than 5

    user_recommendations = {}
    for i, score in sim_scores_user:
        restaurant = userRatings.index[i]
        user_recommendations[restaurant] = score

    return user_recommendations

# Fungsi untuk rekomendasi hibrida
def hybrid_recommendations(user, restaurant_name):
    # Rekomendasi berbasis konten
    content_recommendations = get_content_recommendations(restaurant_name)

    # Rekomendasi kolaboratif
    user_recommendations = get_collaborative_recommendations(user)

    # Gabungkan rekomendasi berbasis konten dan kolaboratif
    hybrid_scores = {}

    # Tambahkan rekomendasi berbasis konten dengan bobot
    for restaurant in content_recommendations:
        hybrid_scores[restaurant] = 0.7  # Bobot dapat disesuaikan

    # Tambahkan rekomendasi kolaboratif dengan bobot
    for restaurant, score in user_recommendations.items():
        if restaurant in hybrid_scores:
            # Jika restoran sudah ada dalam rekomendasi berbasis konten, tambahkan skornya
            hybrid_scores[restaurant] += 0.3 * score  # Bobot dapat disesuaikan
        else:
            # Jika restoran belum ada dalam rekomendasi berbasis konten, tambahkan dengan bobot
            hybrid_scores[restaurant] = 0.3 * score  # Bobot dapat disesuaikan

    # Sort dan dapatkan rekomendasi teratas
    hybrid_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return [restaurant for restaurant, score in hybrid_recommendations]

# Create a Flask web application
app = Flask(__name__)


# Define a route for the home page
@app.route('/')
def home():
    # Get top 10 rated restaurants
    top_rated_restaurants = get_top_rated_restaurants()

    return render_template('index.html', top_rated_restaurants=top_rated_restaurants)

# Function to get top 10 rated restaurants
def get_top_rated_restaurants():
    top_ratings = restaurant_review.groupby('Restaurant')['Rating'].mean().sort_values(ascending=False).head(10)
    top_rated_restaurants = [f"{restaurant} : {rating:.2f}" for restaurant, rating in top_ratings.items()]
    return top_rated_restaurants

# Define a route for handling recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    if request.method == 'POST':
        user = request.form['user']
        restaurant_name = request.form['restaurant']

        # Use your hybrid recommendation function here
        hybrid_recs = hybrid_recommendations(user, restaurant_name)

        # Pass restaurant_metadata to the template
        return render_template('recommendations.html', recommendations=hybrid_recs, restaurant_metadata=restaurant_metadata)

if __name__ == '__main__':
    app.run(debug=True)

