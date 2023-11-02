import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import RidgeCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class SppStackingForecastor(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppStackingForecastor"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        r1 = RidgeCV(alphas=[1e-2, 1e-1], cv=5)
        r2 = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], min_impurity_decrease=0.00000001)
        r4 = XGBRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], learning_rate=0.001, booster='gbtree', tree_method='approx')
        dtr = StackingRegressor(estimators=[('rf', r2), ('xgb', r4)], final_estimator=r1, passthrough=False, n_jobs=-1)
        dtr.fit(train_features, train_labels)
        return dtr