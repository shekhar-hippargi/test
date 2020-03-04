from CollaborativeFiltering.scripts.MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter


class SimpleUserCF:
    def __init__(self, moviesFile, ratingsFile, userId):
        self.moviesFile = moviesFile
        self.ratingsFile = ratingsFile
        self.userId = userId
        self.k = 10

    def recommendMoviesUserCF(self):
        # Load our data set and compute the user similarity matrix
        ml = MovieLens(self.moviesFile, self.ratingsFile)
        data = ml.loadMovieLensLatestSmall()
        trainSet = data.build_full_trainset()
        sim_options = {'name': 'cosine', 'user_based': True}

        model = KNNBasic(sim_options=sim_options)
        model.fit(trainSet)
        simsMatrix = model.compute_similarities()

        # Get top N similar users to our test subject
        testUserInnerID = trainSet.to_inner_uid(self.userId)
        similarityRow = simsMatrix[testUserInnerID]

        similarUsers = []
        for innerID, score in enumerate(similarityRow):
            if innerID != testUserInnerID:
                similarUsers.append((innerID, score))

        kNeighbors = heapq.nlargest(self.k, similarUsers, key=lambda t: t[1])

        # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
        candidates = defaultdict(float)
        for similarUser in kNeighbors:
            innerID = similarUser[0]
            userSimilarityScore = similarUser[1]
            theirRatings = trainSet.ur[innerID]
            for rating in theirRatings:
                candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

        # Build a dictionary of stuff the user has already seen
        watched = {}
        for itemID, rating in trainSet.ur[testUserInnerID]:
            watched[itemID] = 1

        # Get top-rated items from similar users:
        pos = 0
        output = []
        for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not itemID in watched:
                movieID = trainSet.to_raw_iid(itemID)
                output.append((ml.getMovieName(int(movieID)), ratingSum))
                pos += 1
                if pos >= 10:
                    break

        return output
