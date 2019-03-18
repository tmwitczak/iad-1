import numpy as np


class KMeansAlgorithm:
    def __init__(self, kValue, tolerance, maxNumOfIter):
        self.kValue = kValue
        self.tolerance = tolerance
        self.maxNumOfIter = maxNumOfIter

    def Euclidean_distance(firstPoint, secondPoint):
        distance = np.ligalg.norm(firstPoint, secondPoint)
        return distance

