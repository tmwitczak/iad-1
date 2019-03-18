import pandas as pd
import numpy


class InputData:
    def readFile(self, filename):
        if filename == "data\_iris.data":
            data = pd.read_csv(filename, header=None).iloc[:, 0:4].values
        elif filename == "data\_abalone.data":
            data = pd.read_csv(filename, header=None).iloc[:, 1:9].values
        elif filename == "data\_wine.data":
            data = pd.read_csv(filename, header=None).iloc[:, 0:9].values
        else:
            data = pd.read_csv(filename, header=None)
        return data
