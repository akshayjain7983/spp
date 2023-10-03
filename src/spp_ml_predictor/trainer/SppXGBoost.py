import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from xgboost import XGBRegressor

class SppXGBoost(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppXGBoost"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        dtr = XGBRegressor(n_estimators=1000, max_depth=train_features.shape[1], random_state=train_features.shape[1]
                           , learning_rate=0.1, booster='gbtree', n_jobs=-1, tree_method='approx')
        dtr.fit(train_features, train_labels, verbose=False)
        return dtr