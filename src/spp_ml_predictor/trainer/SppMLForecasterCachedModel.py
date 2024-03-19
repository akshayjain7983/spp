import pandas as pd
from datetime import datetime, timedelta
from ..trainer.SppMLForecaster import SppMLForecaster

class SppMLForecasterCachedModel(SppMLForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)


    def __getRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        model = None
        modelLastTrainingDate = None
        modelCache = None
        modelCacheKey = None
        if(self.ctx['cacheAndRetrainModel']):
            modelCache = self.ctx['modelCache']
            mode = self.ctx['mode']
            if(mode == 'index'):
                modelCacheKey = self.ctx['index']
                modelCacheForIndex = modelCache.get(modelCacheKey)
                if(modelCacheForIndex == None):
                    modelCacheForIndex = {}
                    modelCache[modelCacheKey] = modelCacheForIndex

                model = modelCacheForIndex.get('modelInUse')
                modelLastTrainingDate = modelCacheForIndex.get('modelInUseLastTrainingDate')
            else:
                modelCacheKey = self.trainingDataPdf.iloc[0]['exchange_code']
                modelCacheForSecurity = modelCache.get(modelCacheKey)
                if (modelCacheForSecurity == None):
                    modelCacheForSecurity = {}
                    modelCache[modelCacheKey] = modelCacheForSecurity

                model = modelCacheForSecurity.get('modelInUse')
                modelLastTrainingDate = modelCacheForSecurity.get('modelInUseLastTrainingDate')


        train_features_next = train_features.copy()
        train_labels_next = train_labels.copy()

        if (modelLastTrainingDate != None):
            newDateRange = pd.date_range(start=modelLastTrainingDate, end=train_features_next.index[train_features_next.shape[0] - 1], inclusive="right")
            train_features_next = train_features_next.reindex(newDateRange)
            train_labels_next = train_labels_next.reindex(newDateRange)


        if(model == None):
            model = self.__getNewRegressor__(train_features, train_labels)
        else:
            model = self.__getRetrainRegressor__(model, modelLastTrainingDate, train_features, train_labels, train_features_next, train_labels_next)

        if (self.ctx['cacheAndRetrainModel']):
            modelCache[modelCacheKey]['modelInUse'] = model
            modelCache[modelCacheKey]['modelInUseLastTrainingDate'] = train_features_next.index[train_features_next.shape[0]-1]

        return model

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        return None

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):
        return None
