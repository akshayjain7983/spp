import pandas as pd
from datetime import datetime, timedelta
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class SppRandomForests(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppRandomForests"

    def __getName__(self):
        return self.name

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        # param_grid = [{'n_estimators':[100, 200]
        #                 , 'min_impurity_decrease':[1e-7, 1e-8, 1e-9]
        #                 , 'max_depth':[int(train_features.shape[1]/2), train_features.shape[1], train_features.shape[1]*2]
        #                 , 'random_state':[train_features.shape[1]]
        #                 , 'warm_start':[True]
        #                 }]
        #
        # grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
        # grid_search.fit(train_features, train_labels)
        model = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], n_jobs=-1, warm_start=True)
        model.fit(train_features, train_labels)
        # model = grid_search.best_estimator_
        return model

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):

        model.n_estimators += 1
        model.fit(train_features_next, train_labels_next)
        return model