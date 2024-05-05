import pandas as pd
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam
from datetime import datetime, date, timedelta
import numpy as np
from ..util import util

class SppLSTM(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppLSTM"

    def __getName__(self):
        return self.name

    def __setupTrainingData__(self) -> pd.DataFrame:
        trainingData = self.trainingDataPdf[['value']]
        trainingData.sort_index(inplace=True)
        return trainingData


    def __buildAndForecast__(self, trainingData:pd.DataFrame) -> pd.DataFrame:

        holidays = self.ctx['holidays']
        endDate = self.ctx['trainingEndDate']
        endDate = util.previous_business_date(endDate, 1, holidays)
        nextForecastDate = util.next_business_date(endDate, 1, holidays)
        nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
        forecastDays = self.ctx['forecastDays']
        trainingData.dropna(inplace=True)
        train = trainingData[:endDate].copy()
        pred:pd.DataFrame = trainingData[nextForecastDate:].copy()
        xtraDataPdfTrain = self.xtraDataPdf[self.xtraDataPdf.index.isin(train.index)]
        train = pd.concat([train, xtraDataPdfTrain], axis=1)
        train['features'] = train[train.columns].apply(tuple, axis=1).apply(list)
        train['features'] = train.features.apply(lambda x: [list(x)])
        train['cumulative_features'] = train.features.cumsum()

        # dtr = self.__getRegressor__(train_features, train_labels)
        dtr = None
        for i in range(forecastDays+1):
            pred_features = pred.filter(items=[nextForecastDateTimeIndex], axis=0)[[f'value_lag_log_{i+1}' for i in range(self.lags)]]
            xtraDataPdfPred = self.xtraDataPdf[self.xtraDataPdf.index.isin(pred_features.index)]
            pred_features = pd.concat([pred_features, xtraDataPdfPred], axis=1)
            pred_features = self.__preparePredFeatures__(pred_features)
            pred_label = self.__predict__(dtr, pred_features)
            nextForecastValueLagLog = pred_label
            nextForecastValue = np.exp(nextForecastValueLagLog)
            pred.loc[nextForecastDateTimeIndex, 'value_lag_0'] = nextForecastValue
            pred.loc[nextForecastDateTimeIndex, 'value_lag_log_0'] = nextForecastValueLagLog
            thisForecastDate = nextForecastDate
            nextForecastDate = util.next_business_date(thisForecastDate, 1, holidays)
            nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
            nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate)
            pred.loc[nextForecastDateTimeIndex] = nextRow

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'd', "forecastDate": pred.index[d].date(), "value": pred['value_lag_0'].iloc[d]} for d in [0, forecastDays]]
        }

        return pd.DataFrame(forecastValues, index=[0, forecastDays])