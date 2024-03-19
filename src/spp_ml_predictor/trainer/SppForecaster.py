
import pandas as pd

class SppForecaster:
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.trainingDataPdf = trainingDataPdf
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def __getName__(self):
        return None

    def forecast(self) -> pd.DataFrame:
        pass