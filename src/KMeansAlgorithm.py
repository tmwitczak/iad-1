import matplotlib.pyplot as plt

# ////////////////////////////////////////// To delete
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans


# //////////////////////////////////////////


class KMeansAlgorithm:
    def __init__(self, kValue, tolerance, maxNumOfIter):
        self.kValue = kValue
        self.tolerance = tolerance
        self.maxNumOfIter = maxNumOfIter
