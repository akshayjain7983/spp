import pandas as pd
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
from ..util import util
import glob, os
from sklearn.model_selection import train_test_split

class SppNN(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppNN"

    def __getName__(self):
        return self.name

    def __setupTrainingData__(self) -> pd.DataFrame:
        trainingData = self.trainingDataPdf[['value']]
        trainingData.sort_index(inplace=True)
        trainingData.rename(columns={"value":"value_lag_0"}, inplace=True)
        concatLags = (trainingData, )
        for i in range(self.lags):
            concatLags = concatLags + (trainingData['value_lag_0'].shift(i+1).rename(f'value_lag_{i+1}'), )

        trainingData = pd.concat(concatLags, axis=1, copy=False)

        concatLogs = (trainingData, )
        for i in range(self.lags+1):
            concatLogs = concatLogs + (np.log(trainingData[f'value_lag_{i}']).rename(f'value_lag_log_{i}'), )

        trainingData = pd.concat(concatLogs, axis=1, copy=False)

        xtraDataPdfTrain = self.xtraDataPdf[self.xtraDataPdf.index.isin(trainingData.index)]
        xtraDataPdfTrain.sort_index(inplace=True)
        trainingData = pd.concat([trainingData, xtraDataPdfTrain], axis=1)

        concatCsm = (trainingData, )
        for i in range(self.candlestickLags):
            concatCsm = concatCsm + (trainingData['candlestickMovement'].shift(i+1).rename(f'candlestickMovementLag{i+1}'), )

        trainingData = pd.concat(concatCsm, axis=1, copy=False)
        trainingData.rename(columns={"candlestickMovement":"candlestickMovementLag0"}, inplace=True)

        return trainingData

    def __buildAndForecast__(self, trainingData:pd.DataFrame) -> pd.DataFrame:

        holidays = self.ctx['holidays']
        endDate = self.ctx['trainingEndDate']
        trainingData.dropna(inplace=True)
        train = trainingData[:endDate].copy()
        endDate = util.previous_business_date(endDate, 1, holidays)
        nextForecastDate = util.next_business_date(endDate, 1, holidays)
        nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
        forecastDays = self.ctx['forecastDays']
        pred:pd.DataFrame = trainingData[nextForecastDate:].copy()
        train_features_cols = np.concatenate(([f'value_lag_log_{i + 1}' for i in range(self.lags)],
                                              ['repo', 'inflation'],
                                              [f'candlestickMovementLag{i + 1}' for i in range(self.candlestickLags)]
                                              ))
        train_features = train[train_features_cols]
        train_labels = train['value_lag_log_0']
        previousValue = train['value_lag_0'].iloc[-1]

        dtr = self.__getRegressor__(train_features, train_labels)

        for i in range(forecastDays*2+1):
            if(i > 0): #no need to predict day 0
                pred_features = pred.filter(items=[nextForecastDateTimeIndex], axis=0)[train_features_cols]
                pred_features = self.__preparePredFeatures__(pred_features)
                pred_label = self.__predict__(dtr, pred_features)
                nextForecastValueLagLog = pred_label
                nextForecastValue = np.exp(nextForecastValueLagLog)
                candleStickRealBodyChangeLag0 = util.candlestick_movement(previousValue, nextForecastValue) #assuming previous close is today's open
                pred.loc[nextForecastDateTimeIndex, 'value_lag_0'] = nextForecastValue
                pred.loc[nextForecastDateTimeIndex, 'value_lag_log_0'] = nextForecastValueLagLog
                pred.loc[nextForecastDateTimeIndex, 'candlestickMovementLag0'] = candleStickRealBodyChangeLag0
            else:
                nextForecastValue = pred.at[nextForecastDateTimeIndex, 'value_lag_0']

            thisForecastDate = nextForecastDate
            nextForecastDate = util.next_business_date(thisForecastDate, 1, holidays)
            nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
            nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate, nextForecastDate)
            pred.loc[nextForecastDateTimeIndex] = nextRow
            previousValue = nextForecastValue

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'd', "forecastDate": pred.index[d].date(), "value": pred['value_lag_0'].iloc[d]} for d in [forecastDays, forecastDays*2]]
        }

        return pd.DataFrame(forecastValues, index=[forecastDays, forecastDays*2])

    def __getNextRow__(self, pred:pd.DataFrame, nextForecastValue:float, thisForecastDate:datetime, nextForecastDate:datetime):
        row = []
        thisForecastDateTimeIndex = datetime.combine(thisForecastDate, datetime.min.time())
        nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
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

        if(nextForecastDateTimeIndex in self.xtraDataPdf.index):
            row.append(self.xtraDataPdf.at[nextForecastDateTimeIndex, 'repo'])
            row.append(self.xtraDataPdf.at[nextForecastDateTimeIndex, 'inflation'])
        else:
            row.append(0.0)
            row.append(0.0)

        for i in range(self.candlestickLags + 1):
            if (i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'candlestickMovementLag{i - 1}'][thisForecastDateTimeIndex])

        return row

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        x, x_test, y, y_test = train_test_split(train_features, train_labels, test_size=0.2, shuffle=False)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, shuffle=False)
        rmse = 100
        count = 1
        model = None
        while(rmse > 0.005 and count < 5):
            modelTemp = Sequential()
            modelTemp.add(Dense(512, activation='relu'))
            modelTemp.add(Dense(512, activation='relu'))
            modelTemp.add(Dense(1))
            opt = Adam(learning_rate=0.00001)
            modelTemp.compile(optimizer=opt, loss='mean_squared_error', metrics=['root_mean_squared_error'])
            batch_size, epochs = self.__getBatchSizeEpochs__(x)
            modelTemp.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid), verbose=0)
            mse_test = modelTemp.evaluate(x_test, y_test, verbose=0)
            rmseTemp = mse_test[1]
            if(rmseTemp < rmse):
                rmse = rmseTemp
                model = modelTemp
            count += 1
        return model

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):
        opt = Adam(learning_rate=0.000001)
        model.compile(optimizer=opt, loss='mean_squared_error')
        batch_size, epochs = self.__getBatchSizeEpochs__(train_features_next)
        model.fit(train_features_next, train_labels_next, batch_size=batch_size, epochs=epochs, verbose=0)
        return model

    def __getBatchSizeEpochs__(self, train_features):
        default_batch_size = self.lags * 3
        batch_size = default_batch_size if (len(train_features) >= default_batch_size) else len(train_features)
        epochs = int(len(train_features) / batch_size) + 1
        epochs = epochs if(epochs <= 50) else 50
        return batch_size, epochs

    def __predict__(self, regressor, pred_features:pd.DataFrame):
        return regressor.predict(pred_features, verbose=0)[0][0]

    def __load_model__(self):
        fileName = self.__find_model_filename()
        if(fileName):
            model = load_model(fileName)
            modelInUseLastTrainingDate = self.__find_modelInUseLastTrainingDate__(fileName)
            return model, modelInUseLastTrainingDate
        else:
            return super().__load_model__()
    def __find_modelInUseLastTrainingDate__(self, fileName:str):
        dateStr = fileName.split('.')[0].split('__')[3]
        return datetime.strptime(dateStr, '%Y-%m-%d')

    def __find_model_filename(self):
        fileName = self.__build_model_filename(None)
        arr = glob.glob(fileName, root_dir=self.ctx['config']['ml-models.location'])
        if (len(arr) == 0):
            return None

        arr.sort()
        fileName = arr[-1]
        return fileName

    def __persist_model__(self, modelCacheForKey: dict):
        fileName = self.__find_model_filename()
        if (fileName):
            os.remove(fileName)

        modelInUse = modelCacheForKey['modelInUse']
        modelInUseLastTrainingDate = modelCacheForKey['modelInUseLastTrainingDate']
        fileName = self.__build_model_filename(modelInUseLastTrainingDate)
        modelInUse.save(fileName)

    def __build_model_filename(self, modelInUseLastTrainingDate):
        mode = self.ctx['mode']
        fileName = self.ctx['config']['ml-models.location']
        fileName += 'SPP-ML-Model__' + self.__getName__() + '__';
        if (mode == 'index'):
            fileName += self.ctx['index'] + '__'
        else:
            fileName += self.trainingDataPdf.iloc[0]['exchange_code'] + '__'

        if (modelInUseLastTrainingDate != None):
            fileName += datetime.strftime(modelInUseLastTrainingDate, '%Y-%m-%d')
        else:
            fileName += '*'

        fileName += '.keras'

        return fileName
