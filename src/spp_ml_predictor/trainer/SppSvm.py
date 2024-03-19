import pandas as pd
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class SppSvm(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppSvm"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        param_grid = [
            {'kernel':['poly', 'rbf'], 'degree': [3, 5, 10], 'epsilon': [1, 2, 4], 'C':[0.1, 2, 4], 'tol':[1e-7, 1e-8, 1e-10]}
        ]

        poly_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', refit='neg_mean_squared_error'))
        ])
        poly_kernel_svm_clf.fit(train_features, train_labels)
        return poly_kernel_svm_clf