import pandas as pd


class InputData:
    def readFile(self):
        irisData = pd.read_csv("iris.data", header=None)
