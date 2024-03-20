import pandas as pd
from datetime import datetime, date, timedelta
from ..trainer.SppForecaster import SppForecaster
import numpy as np

class SppMLForecaster(SppForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.lags = 15


    def forecast(self) -> pd.DataFrame:
        trainingData = self.__setupTrainingData__()
        forecastData = self.__buildAndForecast__(trainingData)
        return forecastData


    def __setupTrainingData__(self) -> pd.DataFrame:
        trainingData = self.trainingDataPdf[['value']]

        startDate = self.ctx['trainingStartDate']
        endDate = self.ctx['trainingEndDate']
        nextForecastDate = endDate + timedelta(days=1)
        trainingDataReindexPdf = pd.date_range(start=startDate, end=nextForecastDate)
        trainingData = trainingData.reindex(trainingDataReindexPdf)
        trainingData.sort_index(inplace=True)

        trainingData.rename(columns={"value":"value_lag_0"}, inplace=True)

        for i in range(self.lags):
            trainingData[f'value_lag_{i+1}'] = trainingData['value_lag_0'].shift(i+1)

        for i in range(self.lags+1):
            trainingData[f'value_lag_log_{i}'] = np.log(trainingData[f'value_lag_{i}'])

        for i in range(self.lags):
            trainingData[f'value_lag_log_diff_{i+1}'] = trainingData[f'value_lag_log_{i}'] - trainingData[f'value_lag_log_{i+1}']

        return trainingData

    def __getRegressor__(self, train_features:pd.DataFrame, train_labels:pd.DataFrame):
        return None

    def __preparePredFeatures__(self, pred_features:pd.DataFrame):
        return pred_features

    def __predict__(self, regressor, pred_features:pd.DataFrame):
        return regressor.predict(pred_features)

    def __buildAndForecast__(self, trainingData:pd.DataFrame) -> pd.DataFrame:

        endDate = self.ctx['trainingEndDate']
        endDate = endDate - timedelta(days=1)
        nextForecastDate = endDate + timedelta(days=1)
        nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
        forecastDays = self.ctx['forecastDays']
        train = trainingData[:endDate].copy()
        pred:pd.DataFrame = trainingData[nextForecastDate:].copy()
        train = train.dropna()
        xtraDataPdfTrain = self.xtraDataPdf[self.xtraDataPdf.index.isin(train.index)]
        train_features = train[[f'value_lag_log_diff_{i+2}' for i in range(self.lags-1)]]
        train_features = pd.concat([train_features, xtraDataPdfTrain], axis=1)
        train_labels = train['value_lag_log_diff_1']

        dtr = self.__getRegressor__(train_features, train_labels)

        for i in range(forecastDays[-1]+1):
            pred_features = pred.filter(items=[nextForecastDateTimeIndex], axis=0)[[f'value_lag_log_diff_{i+2}' for i in range(self.lags-1)]]
            xtraDataPdfPred = self.xtraDataPdf[self.xtraDataPdf.index.isin(pred_features.index)]
            pred_features = pd.concat([pred_features, xtraDataPdfPred], axis=1)
            pred_features = self.__preparePredFeatures__(pred_features)
            pred_labels = self.__predict__(dtr, pred_features)
            nextForecastValueLagLogDiff1 = pred_labels[0]
            nextForecastValueLagLog1 = pred['value_lag_log_1'][nextForecastDateTimeIndex]
            nextForecastValueLagLog = nextForecastValueLagLog1 + nextForecastValueLagLogDiff1
            nextForecastValue = np.exp(nextForecastValueLagLog)
            pred['value_lag_0'][nextForecastDateTimeIndex] = nextForecastValue
            pred['value_lag_log_0'][nextForecastDateTimeIndex] = nextForecastValueLagLog
            pred['value_lag_log_diff_1'][nextForecastDateTimeIndex] = nextForecastValueLagLogDiff1
            thisForecastDate = nextForecastDate
            nextForecastDate = thisForecastDate + timedelta(days=1)
            nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
            nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate)
            pred.loc[nextForecastDateTimeIndex] = nextRow

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'd', "forecastDate": pred.index[d].date(), "value": pred['value_lag_0'].iloc[d]} for d in forecastDays]
        }

        return pd.DataFrame(forecastValues, index=forecastDays)

    def __getNextRow__(self, pred:pd.DataFrame, nextForecastValue:float, thisForecastDate:datetime):
        row = []
        thisForecastDateTimeIndex = datetime.combine(thisForecastDate, datetime.min.time())
        for i in range(self.lags+1):
            if(i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'value_lag_{i-1}'][thisForecastDateTimeIndex])

        for i in range(self.lags+1):
            if(i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'value_lag_log_{i-1}'][thisForecastDateTimeIndex])

        for i in range(self.lags):
            if(i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'value_lag_log_diff_{i}'][thisForecastDateTimeIndex])

        return row