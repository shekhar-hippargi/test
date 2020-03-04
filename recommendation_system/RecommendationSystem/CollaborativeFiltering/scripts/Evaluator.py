from CollaborativeFiltering.scripts.EvaluationData import EvaluationData
from CollaborativeFiltering.scripts.EvaluatedAlgorithm import EvaluatedAlgorithm


class Evaluator:
    algorithms = []

    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed

    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

    def Evaluate(self, doTopN):
        results = {}
        # output = ''
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results
        print("\n")

        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            # output = output + "\n" + "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            #     "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty")
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
                # output = output + "\n" + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                #     name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                #     metrics["Coverage"], metrics["Diversity"], metrics["Novelty"])
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            # output = output + "\n" + "{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE")
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                # output = output + "\n" + "{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"])

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        # output = output + "\nLegend:\n"
        # output = output + "\nRMSE:      Root Mean Squared Error. Lower values mean better accuracy."
        # output = output + "\nMAE:       Mean Absolute Error. Lower values mean better accuracy."
        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print(
                "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
            print(
                "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print(
                "Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")
            # output = output + "\n" + "HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better."
            # output = output + "\n" + "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better."
            # output = output + "\n" + "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better."
            # output = output + "\n" + "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better."
            # output = output + "\n" + "Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations " \
            #                          "for a given user. Higher means more diverse."
            # output = output + "\n" + "Novelty:   Average popularity rank of recommended items. Higher means more novel."
            # return output
    def SampleTopNRecs(self, ml, userId, k=10):
        # output = ''
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            # output = output + "\nUsing recommender ", str(algo.GetName())
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(userId)

            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            print("\nWe recommend:")
            # output = output + "\nRecommended Movies are:"
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                # output = output + str(ml.getMovieName(ratings[0]), ratings[1])
                print(ml.getMovieName(ratings[0]), ratings[1])
