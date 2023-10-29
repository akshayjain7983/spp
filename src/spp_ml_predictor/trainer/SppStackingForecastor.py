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

        # dtr1 = XGBRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1]
        #                    , learning_rate=0.5, booster='gbtree', tree_method='approx')
        # dtr2 = SVR(kernel="poly", degree=10, C=0.1, epsilon=0.1)

        r1 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1], cv=5)
        r2 = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], min_impurity_decrease=0.00000001)
        r3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=train_features.shape[1], splitter='best', random_state=train_features.shape[1], min_impurity_decrease=0.00000001), n_estimators=500, learning_rate=0.1)
        r4 = XGBRegressor(n_estimators=1000, max_depth=train_features.shape[1], random_state=train_features.shape[1], learning_rate=0.1, booster='gbtree', tree_method='approx')

        # rf1 = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=10, min_impurity_decrease=0.00001)
        # rf2 = RandomForestRegressor(n_estimators=25, max_depth=12, random_state=11, min_impurity_decrease=0.000001)
        # rf3 = RandomForestRegressor(n_estimators=25, max_depth=24, random_state=12, min_impurity_decrease=0.0000001)
        # rf4 = RandomForestRegressor(n_estimators=25, max_depth=48, random_state=13, min_impurity_decrease=0.00000001)
        dtr = StackingRegressor(estimators=[('r2', r2), ('r4', r4)], final_estimator=r1, passthrough=True, n_jobs=-1)
        dtr.fit(train_features, train_labels)
        return dtr