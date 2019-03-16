from InputData import InputData
from KMeansAlgorithm import KMeansAlgorithm


def main():
    # Inputing data
    firstDataSet = InputData("iris.data")
    irisData = firstDataSet.readFile()

    secondDataSet = InputData("abalone.data")
    abaloneData = secondDataSet.readFile()

    thirdDataSet = InputData("seeds.data")
    seedsData = thirdDataSet.readFile()

    kMeans = KMeansAlgorithm()
    kMeans.imageCompressionFromTutorial()


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == "__main__":
    main()
# /////////////////////////////////////////////////////////////////////////// #
