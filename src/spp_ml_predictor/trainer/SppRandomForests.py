import pandas as pd
from datetime import datetime, timedelta
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from sklearn.ensemble import RandomForestRegressor

class SppRandomForests(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppRandomForests"
        self.maxEstimators = 300

    def __getName__(self):
        return self.name

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        model = RandomForestRegressor(n_estimators=int(self.maxEstimators*2/3), max_depth=train_features.shape[1], random_state=train_features.shape[1], n_jobs=-1, warm_start=True)
        model.fit(train_features, train_labels)
        return model

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):

        model.n_estimators += 1
        model.fit(train_features_next, train_labels_next)
        return model