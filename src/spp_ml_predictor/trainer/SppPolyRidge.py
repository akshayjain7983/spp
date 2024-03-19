import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..trainer.SppMLForecaster import SppMLForecaster

class SppPolyRidge(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppPolyRidge"
        self.poly_features = Pipeline([
            ("poly_features", PolynomialFeatures(degree=(1, 3), include_bias=False)),
            ("std_scaler", StandardScaler())
        ])

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        train_features_poly = self.poly_features.fit_transform(train_features, train_labels)
        dtr = Ridge(alpha=0.00001, solver="auto")
        dtr.fit(train_features_poly, train_labels)
        return dtr

    def __preparePredFeatures__(self, pred_features:pd.DataFrame):
        pred_features_poly = self.poly_features.transform(pred_features)
        return pred_features_poly
