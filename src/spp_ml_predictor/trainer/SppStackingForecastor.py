import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

class SppStackingForecastor(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppStackingForecastor"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        dtr1 = XGBRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1]
                           , learning_rate=0.5, booster='gbtree', tree_method='approx')
        dtr2 = RidgeCV()
        dtr3 = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], min_impurity_decrease=0.00001)
        dtr = StackingRegressor(estimators=[('gb', dtr1), ('rf', dtr3)], final_estimator=dtr2, n_jobs=-1)
        dtr.fit(train_features, train_labels)
        return dtr