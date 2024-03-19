import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from ..trainer.SppMLForecaster import SppMLForecaster

class SppDecisionTree(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppDecisionTree"

    def __getName__(self):
        return self.name

    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        dtr = DecisionTreeRegressor(max_depth=train_features.shape[1])
        dtr.fit(train_features, train_labels)
        return dtr


