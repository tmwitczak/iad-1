import pandas as pd
import numpy


class InputData:
    def readFile(self, filename):
        if filename == "data\_iris.data":
            data = pd.read_csv(filename, header=None, sep=",")
        elif filename == "data\_abalone.data":
            data = pd.read_csv(filename, header=None, sep=",")
        elif filename == "data\_seeds.data":
            data = pd.read_csv(filename, header=None, sep=" ")
        return data
