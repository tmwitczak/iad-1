import math


class KMeansAlgorithm:
    def __init__(self, kValue, tolerance, maxNumOfIter):
        self.kValue = kValue
        self.tolerance = tolerance
        self.maxNumOfIter = maxNumOfIter

    # def Euclidean_distance(feat_one, feat_two):
    #     squared_distance = 0
    #
    #     # Assuming correct input to the function where the lengths of two features are the same
    #
    #     for i in range(len(feat_one)):
    #         squared_distance += (feat_one[i] – feat_two[i]) ** 2
    #
    #     ed = sqrt(squared_distances)
    #
    #     return ed
