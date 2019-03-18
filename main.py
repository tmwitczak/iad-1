from src.InputData import InputData
from src.KMeansAlgorithm import KMeansAlgorithm
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd


def main():
    # Inputing data
    inputData = InputData()
    irisData = inputData.readFile("data\_iris.data")
    abaloneData = inputData.readFile("data\_abalone.data")
    wineData = inputData.readFile("data\_wine.data")

    # # inputing parameters from user
    # kValue = input("Podaj wartosc K: ")
    # tolerance = input("Podaj wartosc tolerancji: ")
    # maxNumOfIter = input("Podaj maksymalna liczbe iteracji: ")
    # kMeans = KMeansAlgorithm(kValue, tolerance, maxNumOfIter)

    # plt.plot(irisData.iloc[:, 0], irisData.iloc[:, 1], "r+", label="Irysy")
    # plt.show()
    # # plt.plot(abaloneData.iloc[:, 2], seedsData.iloc[:, 3], "r+", label="Abalone")
    # # plt.show()
    # plt.plot(wineData.iloc[:, 0], wineData.iloc[:, 1], "r+", label="Wina")
    # plt.show()


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == "__main__":
    main()
# /////////////////////////////////////////////////////////////////////////// #
