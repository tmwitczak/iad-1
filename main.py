from src.InputData import InputData
from src.KMeansAlgorithm import KMeansAlgorithm


def main():
    # Inputing data
    inputData = InputData()
    irisData = inputData.readFile("data\_iris.data")
    abaloneData = inputData.readFile("data\_abalone.data")
    seedsData = inputData.readFile("data\_seeds.data")

    # inputing parameters from user
    kValue = input("Podaj wartosc K: ")
    tolerance = input("Podaj wartosc tolerancji: ")
    maxNumOfIter = input("Podaj maksymalna liczbe iteracji: ")
    kMeans = KMeansAlgorithm(kValue, tolerance, maxNumOfIter)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == "__main__":
    main()
# /////////////////////////////////////////////////////////////////////////// #
