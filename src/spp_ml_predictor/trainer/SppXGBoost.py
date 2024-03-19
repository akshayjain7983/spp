import pandas as pd
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from xgboost import XGBRegressor

class SppXGBoost(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppXGBoost"

    def __getName__(self):
        return self.name

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        model = XGBRegressor(n_estimators=1000, max_depth=train_features.shape[1], random_state=train_features.shape[1]
                           , learning_rate=0.001, booster='gbtree', n_jobs=-1, tree_method='approx')
        model.fit(train_features, train_labels, verbose=False)
        print(self.ctx['mode'])
        print(list(zip(model.feature_importances_, train_features)))
        return model

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):
        model.fit(train_features_next, train_labels_next, verbose=False, xgb_model=model.get_booster())
        return model