from CollaborativeFiltering.scripts.MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter


class SimpleItemCF:
    def __init__(self, moviesFile, ratingsFile, userId):
        self.moviesFile = moviesFile
        self.ratingsFile = ratingsFile
        self.userId = userId
        self.k = 10

    def recommendMovies(self):
        ml = MovieLens(self.moviesFile, self.ratingsFile)
        data = ml.loadMovieLensLatestSmall()

        trainSet = data.build_full_trainset()

        sim_options = {'name': 'cosine',
                       'user_based': False
                       }

        model = KNNBasic(sim_options=sim_options)
        model.fit(trainSet)
        simsMatrix = model.compute_similarities()

        testUserInnerID = trainSet.to_inner_uid(self.userId)

        # Get the top K items we rated
        testUserRatings = trainSet.ur[testUserInnerID]
        kNeighbors = heapq.nlargest(self.k, testUserRatings, key=lambda t: t[1])

        # Get similar items to stuff we liked (weighted by rating)
        candidates = defaultdict(float)
        for itemID, rating in kNeighbors:
            similarityRow = simsMatrix[itemID]
            for innerID, score in enumerate(similarityRow):
                candidates[innerID] += score * (rating / 5.0)

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
