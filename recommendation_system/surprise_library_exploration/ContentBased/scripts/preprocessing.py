# Data frame manipulation library
import pandas as pd

# Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Loading datasets
# Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('../data/movies.csv')

# Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('../data/ratings.csv')

# Preprocessing movies_df

# remove 'year' from title and make new column
# Using regular expressions to find a year stored between parentheses
# We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
# print(movies_df.head())

# Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
# print(movies_df.head())

# Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1


# Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())

# Now Preprocessing ratings_df
ratings_df = ratings_df.drop('timestamp', 1)
# print(ratings_df.head())

# movies_df.to_csv("../preprocessedData/moviesPreprocessed.csv", index=False)

# moviesWithGenres_df.to_csv("../preprocessedData/moviesWithGenres.csv", index=False)

print(moviesWithGenres_df.shape)