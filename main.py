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

    # inputing parameters from user
    kValue = input("Podaj wartosc K: ")
    kMeans = KMeansAlgorithm(kValue)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == "__main__":
    main()
# /////////////////////////////////////////////////////////////////////////// #
