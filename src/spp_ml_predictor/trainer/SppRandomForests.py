from datetime import datetime, timedelta
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as psf

from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from pyspark.ml.pipeline import Pipeline


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
class SppRandomForests(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:ps.DataFrame, ctx:dict, xtraDataPdf:ps.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppRandomForests"

    def __getName__(self):
        return self.name
    
    def __getRegressor__(self, train_features: ps.DataFrame, train_labels: ps.DataFrame):
        # param_grid = [{'n_estimators':[100, 200]
        #                 , 'min_impurity_decrease':[1e-7, 1e-8, 1e-9]
        #                 , 'max_depth':[int(train_features.shape[1]/2), train_features.shape[1], train_features.shape[1]*2]
        #                 , 'random_state':[train_features.shape[1]]
        #                 , 'warm_start':[True]
        #                 }]
    
        # grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
        # grid_search.fit(train_features.pandas_api('date'), train_labels.pandas_api('date'))
        # model = RandomForestRegressor(n_estimators=200, max_depth=len(train_features.dtypes), random_state=len(train_features.dtypes), n_jobs=-1, warm_start=True)
        # model.fit(train_features_panda, train_labels_panda)
        # model = grid_search.best_estimator_
        train_features = train_features.dropna();
        train_labels = train_labels.dropna();
        featureCols = [c[0] for c in train_features.dtypes]
        featureCols.remove('date')
        train_data = train_features.join(train_labels, 'date', 'inner')
        vectorizer = VectorAssembler(inputCols=featureCols, outputCol='features')
        rf = RandomForestRegressor(numTrees=200, maxDepth=min(len(featureCols), 30), labelCol='value_lag_log_diff_1')
        pipeline = Pipeline(stages=[vectorizer, rf])
        model = pipeline.fit(train_data)
        
        return model
    
    # def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
    #
    #     param_grid = [{'n_estimators':[100, 200]
    #                     , 'min_impurity_decrease':[1e-7, 1e-8, 1e-9]
    #                     , 'max_depth':[int(train_features.shape[1]/2), train_features.shape[1], train_features.shape[1]*2]
    #                     , 'random_state':[train_features.shape[1]]
    #                     , 'warm_start':[True]
    #                     }]
    #
    #     grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    #     grid_search.fit(train_features, train_labels)
    #     # model = RandomForestRegressor(n_estimators=200, max_depth=train_features.shape[1], random_state=train_features.shape[1], n_jobs=-1, warm_start=True)
    #     # model.fit(train_features, train_labels)
    #     model = grid_search.best_estimator_
    #     return model
    #
    # def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
    #                             , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):
    #
    #     model.n_estimators += 1
    #     model.fit(train_features_next, train_labels_next)
    #     return model