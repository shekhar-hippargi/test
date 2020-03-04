import pandas as pd

movies_df = pd.read_csv("../preprocessedData/moviesPreprocessed.csv")

# Note: To add more movies, simply increase the amount of elements in the userInput.
# Just be sure to write it in with capital letters
# and if a movie starts with a "The", like "The Matrix" then write it in like this: 'Matrix, The' .
userInput = [{'title': 'Balto', 'rating': 5}]


# userInput = [
#             {'title': 'Breakfast Club, The', 'rating': 5},
#             {'title': 'Toy Story', 'rating': 3.5},
#             {'title': 'Jumanji', 'rating': 2},
#             {'title': "Pulp Fiction", 'rating': 5},
#             {'title': 'Akira', 'rating': 4.5}
#          ]

inputMovies = pd.DataFrame(userInput)
# print(inputMovies)

# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

# Final input dataframe
# If a movie you added in above isn't here, then it might not be in the original
# dataframe or it might spelled differently, please check capitalisation.
print(inputMovies)

moviesWithGenres_df = pd.read_csv("../preprocessedData/moviesWithGenres.csv")

# Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
# print(userMovies)

# Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
# Dropping unnecessary issues to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
# print(userGenreTable)

# Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
# The user profile
print(userProfile)

# Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
# print(genreTable.head())

# Multiplying the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
# print(recommendationTable_df.head())

# The final recommendation table
finalRecommendation = movies_df[['title', 'year']].loc[(movies_df['movieId'].isin(recommendationTable_df.head(20).keys()))]
finalRecommendation.year = finalRecommendation.year.astype(int)
print(finalRecommendation)
