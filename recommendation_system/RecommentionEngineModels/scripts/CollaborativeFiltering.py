import torch
from fastai.learner import *
from fastai.column_data import *
from fastai.imports import *

path = '../Datasets/ml-latest-small/'
ratings = pd.read_csv(path + 'ratings.csv')
print(ratings.head())

movies = pd.read_csv(path + 'movies.csv')
print(movies.head())

g=ratings.groupby('userId')['rating'].count()
topUsers=g.sort_values(ascending=False)[:15]

g=ratings.groupby('movieId')['rating'].count()
topMovies=g.sort_values(ascending=False)[:15]

top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='userId')
top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movieId')

pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)

# Collaborative filtering
val_idxs = get_cv_idxs(len(ratings))
wd = 2e-4
n_factors = 50

cf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating')
learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)
learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2)


