import pandas as pd
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam
from datetime import datetime, date, timedelta
import numpy as np

class SppNN(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppNN"

    def __getName__(self):
        return self.name

    def __setupTrainingData__(self) -> pd.DataFrame:
        trainingData = self.trainingDataPdf[['value']]

        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        nextForecastDate = endDate + timedelta(days=1)
        trainingDataReindexPdf = pd.date_range(start=startDate, end=nextForecastDate, inclusive="both")
        trainingData = trainingData.reindex(trainingDataReindexPdf)
        trainingData.sort_index(inplace=True)

        trainingData.rename(columns={"value":"value_lag_0"}, inplace=True)

        for i in range(self.lags):
            trainingData[f'value_lag_{i+1}'] = trainingData['value_lag_0'].shift(i+1)

        for i in range(self.lags+1):
            trainingData[f'value_lag_log_{i}'] = np.log(trainingData[f'value_lag_{i}'])

        return trainingData

    def __buildAndForecast__(self, trainingData:pd.DataFrame) -> pd.DataFrame:

        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        endDate = endDate - timedelta(days=1)
        nextForecastDate = endDate + timedelta(days=1)
        forecastDays = self.ctx['forecastDays']
        train = trainingData[:endDate].copy()
        pred:pd.DataFrame = trainingData[nextForecastDate:].copy()
        train = train.dropna()
        xtraDataPdfTrain = self.xtraDataPdf[self.xtraDataPdf.index.isin(train.index)]
        train_features = train[[f'value_lag_log_{i+1}' for i in range(self.lags)]]
        train_features = pd.concat([train_features, xtraDataPdfTrain], axis=1)
        train_labels = train['value_lag_log_0']

        dtr = self.__getRegressor__(train_features, train_labels)

        for i in range(forecastDays[-1]+1):
            pred_features = pred.filter(items=[nextForecastDate], axis=0)[[f'value_lag_log_{i+1}' for i in range(self.lags)]]
            xtraDataPdfPred = self.xtraDataPdf[self.xtraDataPdf.index.isin(pred_features.index)]
            pred_features = pd.concat([pred_features, xtraDataPdfPred], axis=1)
            pred_features = self.__preparePredFeatures__(pred_features)
            pred_labels = self.__predict__(dtr, pred_features)
            nextForecastValueLagLog = pred_labels[0]
            nextForecastValue = np.exp(nextForecastValueLagLog)
            pred['value_lag_0'][nextForecastDate] = nextForecastValue
            pred['value_lag_log_0'][nextForecastDate] = nextForecastValueLagLog
            thisForecastDate = nextForecastDate
            nextForecastDate = thisForecastDate + timedelta(days=1)
            nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate)
            pred.loc[nextForecastDate] = nextRow

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'D', "forecastDate": datetime.strftime(pred.index[d], '%Y-%m-%d'), "value": pred['value_lag_0'][d]} for d in forecastDays]
        }

        return pd.DataFrame(forecastValues, index=forecastDays)

    def __getNextRow__(self, pred:pd.DataFrame, nextForecastValue:float, thisForecastDate:datetime):
        row = []
        for i in range(self.lags+1):
            if(i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'value_lag_{i-1}'][thisForecastDate])

        for i in range(self.lags+1):
            if(i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'value_lag_log_{i-1}'][thisForecastDate])

        return row

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):
        model = Sequential()
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.fit(train_features, train_labels, batch_size=1, epochs=1, verbose=0)
        return model

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):
        opt = Adam(learning_rate=0.00001)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.fit(train_features_next, train_labels_next, batch_size=1, epochs=1, verbose=0)
        return model

    def __predict__(self, regressor, pred_features:pd.DataFrame):
        return regressor.predict(pred_features, verbose=0)
