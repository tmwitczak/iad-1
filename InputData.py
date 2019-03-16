import pandas as pd


class InputData:
    def __init__(self, filename):
        self.filename = filename

    def readFile(self):
        data = pd.read_csv(self.filename, header=None)
        return data
