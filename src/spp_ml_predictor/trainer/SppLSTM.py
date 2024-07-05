import pandas as pd
from ..trainer.SppForecaster import SppForecaster
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, date, timedelta
import numpy as np
from ..util import util
from sklearn.model_selection import train_test_split
import keras_tuner as kt


class SppLSTM(SppForecaster):
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.name = "SppLSTM"
        self.lags = self.ctx['forecastDays']*2*3
        self.candlestickLags = self.lags
        self.model_loss_function = "mean_absolute_percentage_error"

    def __getName__(self):
        return self.name

    def forecast(self) -> pd.DataFrame:
        trainingData = self.__setupTrainingData__()
        forecastData = self.__buildAndForecast__(trainingData)
        return forecastData

    def __setupTrainingData__(self) -> pd.DataFrame:
        trainingData = np.log(self.trainingDataPdf[['value']])
        trainingData.sort_index(inplace=True)
        return trainingData

    def Sequential_Input_LSTM(self, df):
        df_np = df.to_numpy()
        X = []
        y = []

        for i in range(len(df_np) - self.lags):
            row = [a for a in df_np[i:i + self.lags]]
            X.append(row)
            label = df_np[i + self.lags]
            y.append(label)

        return np.array(X), np.array(y)

    def __build_model_callable__(self, n_features):

        def __build_model__(hp) -> Sequential:
            modelTemp = Sequential()
            n_neurons = hp.Int('neurons', min_value=256, max_value=1024, step=64)
            n_hidden = hp.Int('hidden_layers', min_value=1, max_value=4, step=1)
            learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6])

            modelTemp.add(InputLayer((self.lags, n_features)))
            n_neurons_layer = n_neurons
            for layer in range(n_hidden):
                modelTemp.add(LSTM(n_neurons_layer, return_sequences=(layer<(n_hidden-1))))
                n_neurons_layer = int(n_neurons_layer/4)

            modelTemp.add(Dense(1))
            opt = Adam(learning_rate=learning_rate)
            modelTemp.compile(optimizer=opt, loss=self.model_loss_function)
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

        tuner = kt.RandomSearch(self.__build_model_callable__(len(x[0][0])),
                            objective=kt.Objective('val_loss', direction='min'),
                            max_trials=10,
                            executions_per_trial=5,
                            overwrite=False,
                            directory=keras_tuner_dir,
                            project_name=keras_tuner_sub_dir)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, shuffle=False)
        stop_early = EarlyStopping(monitor='val_loss', mode='min', min_delta=5, patience=8, start_from_epoch=3)
        tuner.search(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[stop_early])
        return tuner

    def __getBatchSizeEpochs__(self, train_features):
        default_batch_size = self.lags * 3
        batch_size = default_batch_size if (len(train_features) >= default_batch_size) else len(train_features)
        epochs = int(len(train_features) / batch_size) + 1
        epochs = epochs if (epochs <= 50) else 50
        return batch_size, epochs

    def __getNewRegressor__(self, trainingData: pd.DataFrame):
        x, y = self.Sequential_Input_LSTM(trainingData)
        tuner = self.__getTuner__(x, y, 0.2)
        model = tuner.get_best_models()[0]
        return model

    # def __getRetrainRegressor__(self, model, trainingData: pd.DataFrame):
    #     x_train, x_val = train_test_split(trainingData, test_size=0.2, shuffle=False)
    #     window = WindowGenerator(self.lags, 1, 1, x_train, x_val)
    #     batch_size, epoch = self.__getBatchSizeEpochs__(trainingData)
    #     opt = Adam(learning_rate=1e-6)
    #     model.compile(optimizer=opt, loss=self.model_loss_function)
    #     model.fit(window.train, epochs=epoch, validation_data=window.val, batch_size=batch_size)
    #     return model

    def __buildAndForecast__(self, trainingData:pd.DataFrame) -> pd.DataFrame:
        holidays = self.ctx['holidays']
        endDate = self.ctx['trainingEndDate']
        trainingData.dropna(inplace=True)
        train = trainingData[:endDate].copy()
        endDate = util.previous_business_date(endDate, 1, holidays)
        nextForecastDate = util.next_business_date(endDate, 1, holidays)
        nextForecastDateTimeIndex = datetime.combine(nextForecastDate, datetime.min.time())
        forecastDays = self.ctx['forecastDays']
        pred: pd.DataFrame = trainingData[nextForecastDate:].copy()
        model = self.__getNewRegressor__(trainingData)
        predInputData = trainingData[(0-self.lags):]

        print(model)

        forecastValues = {
            "forecastModel": self.__getName__(),
            "forecastValues": [{"forecastPeriod": str(d) + 'd', "forecastDate": pred.index[d].date(), "value": pred['value_lag_0'].iloc[d]} for d in [0, forecastDays]]
        }

        return pd.DataFrame(forecastValues, index=[0, forecastDays])