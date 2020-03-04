from ContentBased.MovieLensData.MovieLens import MovieLens
from ContentBased.MovieLensData.ContentKNNAlgorithm import ContentKNNAlgorithm
from ContentBased.MovieLensData.Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np


def LoadMovieLensData():
    dataset = MovieLens()
    print("Loading movie ratings...")
    data = dataset.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = dataset.getPopularityRanks()
    return dataset, data, rankings


np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
dataset, evaluationData, rankings = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(doTopN=True)

evaluator.SampleTopNRecs(dataset)
