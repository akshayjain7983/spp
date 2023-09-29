import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

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
        dtrRidge = Ridge(alpha=0.0001, solver="auto")
        # dtrRandomForests = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1])
        dtrAdaBoost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=min(train_features.shape[1], 25), splitter='best', random_state=train_features.shape[1]), n_estimators=200, learning_rate=0.5)
        dtr = VotingRegressor(estimators=[('adaBoost', dtrAdaBoost), ('ridge', dtrRidge)], weights=[0.6, 0.4], n_jobs=-1)

        dtr.fit(train_features, train_labels)
        return dtr

    # def __preparePredFeatures__(self, pred_features:pd.DataFrame):
    #     pred_features_poly = self.poly_features.fit_transform(pred_features)
    #     return pred_features_poly