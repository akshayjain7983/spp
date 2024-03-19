import pandas as pd
from sklearn.linear_model import Ridge
from ..trainer.SppMLForecaster import SppMLForecaster

class SppRidge(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppRidge"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        dtr = Ridge(alpha=2, solver="auto")
        dtr.fit(train_features, train_labels)
        return dtr


