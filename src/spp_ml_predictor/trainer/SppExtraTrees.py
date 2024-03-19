import pandas as pd
import numpy as np
from ..trainer.SppMLForecaster import SppMLForecaster
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class SppExtraTrees(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppExtraTrees"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        param_grid = { 'n_estimators':[i for i in range(100, 501, 100)]
                        , 'max_depth':[train_features.shape[1]]
                        , 'max_leaf_nodes':[np.power(2, i) for i in range(4, 9)]
                        , 'random_state':[train_features.shape[1]]
                        , 'min_impurity_decrease':[5/(10**i) for i in range(5, 11)]
                        , 'min_samples_leaf':[5/(10**i) for i in range(1, 6)]
                        , 'min_samples_split':[1/(10**i) for i in range(1, 4)]
                    }

        # dtr = RandomizedSearchCV(ExtraTreesRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', refit='neg_mean_squared_error'
        #                          , n_jobs=-1, random_state=train_features.shape[1])

        dtr = ExtraTreesRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], n_jobs=-1)
        dtr.fit(train_features, train_labels)
        return dtr