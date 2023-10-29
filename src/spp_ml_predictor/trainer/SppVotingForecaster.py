import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNet, SGDRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR

class SppVotingForecaster(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppVotingForecaster"
        # self.poly_features = Pipeline([
        #     ("poly_features", PolynomialFeatures(degree=(1, 3), include_bias=False)),
        #     ("std_scaler", StandardScaler())
        # ])

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        # train_features_poly = self.poly_features.fit_transform(train_features)
        # dtrRidge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1], cv=5)
        # dtrRandomForests = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1])
        # dtrAdaBoost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=min(train_features.shape[1], 25), splitter='best', random_state=train_features.shape[1]), n_estimators=200, learning_rate=0.5)
        # dtrXgBoost = XGBRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1]
        #                    , learning_rate=0.1, booster='gbtree', tree_method='approx')

        r1 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1], cv=5)
        r2 = RandomForestRegressor(n_estimators=500, max_depth=train_features.shape[1], random_state=train_features.shape[1], min_impurity_decrease=0.0000001)
        r3 = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
        r4 = XGBRegressor(n_estimators=500, max_depth=train_features.shape[1], random_state=train_features.shape[1], learning_rate=0.1, booster='gbtree', tree_method='approx')

        # rf1 = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=10, min_impurity_decrease=0.00001)
        # rf2 = RandomForestRegressor(n_estimators=25, max_depth=12, random_state=11, min_impurity_decrease=0.000001)
        # rf3 = RandomForestRegressor(n_estimators=25, max_depth=24, random_state=12, min_impurity_decrease=0.0000001)
        # rf4 = RandomForestRegressor(n_estimators=25, max_depth=48, random_state=13, min_impurity_decrease=0.00000001)
        # dtr = VotingRegressor(estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3), ('rf4', rf4), ('lr', dtr2)], n_jobs=-1)
        dtr = VotingRegressor(estimators=[('r1', r1), ('r2', r2), ('r3', r3), ('r4', r4)], weights=[1, 8, 1, 4], n_jobs=-1)
        dtr.fit(train_features, train_labels)
        return dtr

    # def __preparePredFeatures__(self, pred_features:pd.DataFrame):
    #     pred_features_poly = self.poly_features.fit_transform(pred_features)
    #     return pred_features_poly