
import pandas as pd

class SppForecaster:
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.name = "SppArima"
        self.trainingDataPdf = trainingDataPdf
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def forecast(self) -> pd.DataFrame:
        pass