import os
from _collections import defaultdict
from surprise import BaselineOnly, NormalPredictor, SVD, SVDpp
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# path to dataset file
file_path = os.path.expanduser('../../ContentBased/data/ratings.csv')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)

# # We can now use this dataset as we please, e.g. calling cross_validate
# # cross_validate(BaselineOnly(), data, verbose=True)
# cross_validate(SVD(), data, verbose=True)

# sample random trainset and testset
# test set is made of 25% of the ratings.
# trainset, testset = train_test_split(data, test_size=.25)

# algo = SVD()

# # Train the algorithm on the trainset, and predict ratings for the testset
# algo.fit(trainset)
# predictions = algo.test(testset)
#
# # Then compute RMSE
# accuracy.rmse(predictions)

# predictions = algo.fit(trainset).test(testset)
# predictions = algo.fit(trainset).test(testset)
# Then compute RMSE
# accuracy.rmse(predictions)


trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)



# # get a prediction for specific users and items.
# uid = str(196)
# iid = str(302)
# pred = algo.predict(uid, iid, r_ui=4, verbose=True)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
