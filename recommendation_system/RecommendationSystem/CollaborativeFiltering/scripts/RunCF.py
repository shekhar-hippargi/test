from CollaborativeFiltering.scripts.MovieLens import MovieLens
from surprise import KNNBasic
from surprise import NormalPredictor
from CollaborativeFiltering.scripts.Evaluator import Evaluator

import random
import numpy as np


class RunCF:
    def __init__(self, moviesFile, ratingsFile):
        self.moviesFile = moviesFile
        self.ratingsFile = ratingsFile

    def LoadMovieLensData(self):
        dataset = MovieLens(self.moviesFile, self.ratingsFile)
        print("Loading movie ratings...")
        data = dataset.loadMovieLensLatestSmall()
        print("\nComputing movie popularity ranks so we can measure novelty later...")
        rankings = dataset.getPopularityRanks()
        return dataset, data, rankings

    def recommendMoviesCF(self, userId):
        np.random.seed(0)
        random.seed(0)

        # Load up common data set for the recommender algorithms
        dataset, evaluationData, rankings = self.LoadMovieLensData()

        # Construct an Evaluator to, you know, evaluate them
        evaluator = Evaluator(evaluationData, rankings)

        # User-based KNN
        UserKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
        evaluator.AddAlgorithm(UserKNN, "User KNN")

        # Item-based KNN
        ItemKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
        evaluator.AddAlgorithm(ItemKNN, "Item KNN")

        # Just make random recommendations
        Random = NormalPredictor()
        evaluator.AddAlgorithm(Random, "NormalPredictor")

        evaluator.Evaluate(True)

        evaluator.SampleTopNRecs(dataset, userId)

moviesPath = '../data/ml-latest-small/movies.csv'
ratingsPath = '../data/ml-latest-small/ratings.csv'
obj = RunCF(moviesPath, ratingsPath)
obj.recommendMoviesCF(98)
