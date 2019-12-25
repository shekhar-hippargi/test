import pandas as pd
import numpy as np

ratings = pd.read_csv('/home/shippargi/projects/RecommentionEngineModels/Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=";")

users = pd.read_csv('/home/shippargi/projects/RecommentionEngineModels/Datasets/BX-CSV-Dump/BX-Users.csv', delimiter=";")

books = pd.read_csv('/home/shippargi/projects/RecommentionEngineModels/Datasets/BX-CSV-Dump/BX-Books.csv', delimiter=";", error_bad_lines=False)

print(ratings.shape)
print(users.shape)
print(books.shape)

# Drop 'ImageURLs' in df books
books.drop(['imageURLS','imageURLM','imageURLL'], axis=1, inplace=True)

pd.set_option('display.max_colwidth', -1)
print(ratings['bookRating'].unique())
