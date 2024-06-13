import pandas as pd
from ..trainer.SppMLForecasterCachedModel import SppMLForecasterCachedModel
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import numpy as np
from ..util import util
import glob, os
from sklearn.model_selection import train_test_split

class SppNN(SppMLForecasterCachedModel):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppNN"
        self.model_loss_function = "mean_absolute_error"
        self.model_metrics = ["root_mean_squared_error"]

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

        model = self.__getRegressor__(train_features, train_labels)

        for i in range(forecastDays+1):

            pred_features = pred.filter(items=[nextForecastDateTimeIndex], axis=0)[train_features_cols]
            pred_features = self.__preparePredFeatures__(pred_features)
            pred_label = self.__predict__(model, pred_features)
            nextForecastValueLagLog = pred_label
            nextForecastValue = np.exp(nextForecastValueLagLog)
            candleStickRealBodyChangeLag0 = util.candlestick_movement(previousValue,
                                                                      nextForecastValue)  # assuming previous close is today's open
            pred.loc[nextForecastDateTimeIndex, 'value_lag_0'] = nextForecastValue
            pred.loc[nextForecastDateTimeIndex, 'value_lag_log_0'] = nextForecastValueLagLog
            pred.loc[nextForecastDateTimeIndex, 'candlestickMovementLag0'] = candleStickRealBodyChangeLag0

            thisForecastDate = nextForecastDate
            nextForecastDate = util.next_business_date(thisForecastDate, 1, holidays)
            nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
            nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate, nextForecastDate)
            pred.loc[nextForecastDateTimeIndex] = nextRow
            previousValue = nextForecastValue

        del model

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'd', "forecastDate": pred.index[d].date(), "value": pred['value_lag_0'].iloc[d]} for d in [0, forecastDays]]
        }

        return pd.DataFrame(forecastValues, index=[0, forecastDays])

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
    
    def __build_model_callable__(self):

        def __build_model__(hp:kt.HyperParameters)->Sequential:
         
            modelTemp = Sequential()
            n_neurons = hp.Int('neurons', min_value=256, max_value=2048, step=64)
            n_hidden = hp.Int('hidden_layers', min_value=1, max_value=7, step=1)
            learning_rate = hp.Choice('learning_rate', values=[1e-3/4, 1e-3/2, 1e-3, 1e-4/4, 1e-4/2, 1e-4, 1e-5/4, 1e-5/2, 1e-5, 1e-6/4, 1e-6/2, 1e-6])
            n_neurons_layer = n_neurons
            for layer in range(n_hidden):
                modelTemp.add(Dense(n_neurons_layer, activation='relu'))
                n_neurons_layer = int(n_neurons_layer/2) + 1

            modelTemp.add(Dense(1))
            opt = Adam(learning_rate=learning_rate)
            modelTemp.compile(optimizer=opt, loss=self.model_loss_function, metrics=self.model_metrics)
            return modelTemp
        
        return __build_model__

    def __getTuner__(self, x, y, validation_split=0.0):
        batch_size, epochs = self.__getBatchSizeEpochs__(x)
        root_dir = self.ctx['config']['ml-models.location']
        keras_tuner_dir = root_dir+'keras-tuner/'+self.name+'/'
        mode = self.ctx['mode']
        keras_tuner_sub_dir = None
        if (mode == 'index'):
            keras_tuner_sub_dir = self.ctx['index']
        else:
            keras_tuner_sub_dir = self.trainingDataPdf.iloc[0]['exchange_code']

        monitor = 'val_'+self.model_metrics[0]
        tuner = kt.Hyperband(self.__build_model_callable__(),
                            objective=kt.Objective(monitor, direction='min'),
                            max_epochs=epochs,
                            overwrite=False,
                            directory=keras_tuner_dir,
                            project_name=keras_tuner_sub_dir)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, shuffle=False)
        stop_early = EarlyStopping(monitor=monitor, mode='min', min_delta=0.001, patience=3, start_from_epoch=3)
        tuner.search(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[stop_early])
        return tuner

    def __getNewRegressor__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame):

        tuner = self.__getTuner__(train_features, train_labels, 0.2)
        model = tuner.get_best_models()[0]
        return model

    def __getRetrainRegressor__(self, model, modelLastTrainingDate, train_features: pd.DataFrame, train_labels: pd.DataFrame
                                , train_features_next: pd.DataFrame, train_labels_next: pd.DataFrame):
        
        batch_size, epochs = self.__getBatchSizeEpochs__(train_features_next)
        opt = Adam(learning_rate=1e-6)
        model.compile(optimizer=opt, loss=self.model_loss_function)
        model.fit(train_features_next, train_labels_next, epochs=epochs, batch_size=batch_size, verbose=0)
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
        fileName = self.__find_model_filename__()
        if(fileName):
            model = load_model(fileName)
            modelInUseLastTrainingDate = self.__find_modelInUseLastTrainingDate__(fileName)
            return model, modelInUseLastTrainingDate
        else:
            return super().__load_model__()
    
    def __find_modelInUseLastTrainingDate__(self, fileName:str):
        dateStr = fileName.split('.')[0].split('__')[3]
        return datetime.strptime(dateStr, '%Y-%m-%d')

    def __find_model_filename__(self):
        fileName = self.__build_model_filename__(None)
        arr = glob.glob(fileName, root_dir=self.ctx['config']['ml-models.location'])
        if (len(arr) == 0):
            return None

        arr.sort()
        fileName = arr[-1]
        return fileName

    def __persist_model__(self, modelCache: dict):
        fileName = self.__find_model_filename__()
        if (fileName):
            os.remove(fileName)

        modelInUse = modelCache['modelInUse']
        modelInUseLastTrainingDate = modelCache['modelInUseLastTrainingDate']
        fileName = self.__build_model_filename__(modelInUseLastTrainingDate)
        modelInUse.save(fileName)

    def __build_model_filename__(self, modelInUseLastTrainingDate):
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
