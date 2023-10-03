import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import GradientBoostingRegressor

class SppGradientBoost(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppGradientBoost"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        dtr = GradientBoostingRegressor(max_depth=train_features.shape[1], random_state=train_features.shape[1]
                                        , min_impurity_decrease=0.000005, n_estimators=200, learning_rate=0.5)
        dtr.fit(train_features, train_labels)
        return dtr