import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import RandomForestRegressor

class SppRandomForests(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppRandomForests"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        dtr = RandomForestRegressor(n_estimators=100, max_depth=train_features.shape[1], n_jobs=-1, random_state=train_features.shape[1])
        dtr.fit(train_features, train_labels)
        return dtr