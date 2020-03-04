# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


class RecommendMovies:
    def __init__(self, file):
        self.data = pd.read_csv(file)

        self.C = self.data['vote_average'].mean()
        self.m = self.data['vote_count'].quantile(0.9)

        self.q_movies = self.data.copy().loc[self.data['vote_count'] >= self.m]

        # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a', etc.
        self.tfidf = TfidfVectorizer(stop_words='english')

        # Replace NaN with an empty string
        self.data['overview'] = self.data['overview'].fillna('')

        # Construct the required TF-IDF matrix by fitting and transforming the data
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['overview'])

        # Compute the cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()

    # Function that takes in movie title as input and outputs most similar movies

    def get_recommendations(self, title):
        if title not in self.data['title'].values:
            return 'This movie is not in our database.\nPlease check if you spelled it correct using camel casing'
        else:
            # Get the index of the movie that matches the title
            # if title not in data
            idx = self.indices[title]

            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the 10 most similar movies
            sim_scores = sim_scores[1:11]

            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]

            # Return the top 10 most similar movies
            return self.data['title'].iloc[movie_indices]


# recom = RecommendMovies(file="/home/shippargi/projects/RecommendationSystem/ContentBased/MovieRecommendationTFIDF/data/tmdb_5000_movies.csv")
# print(recom.get_recommendations("Inception"))
