import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.tree import DecisionTreeRegressor

class SppDecisionTree:
    def __init__(self, trainingDataPdf:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.name = "SppDecisionTree"
        self.trainingDataPdf = trainingDataPdf
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf
        self.lags = int(self.ctx['forecastDays']/6)

    def forecast(self) -> pd.DataFrame:
        trainingData = self.__setupTrainingData__()
        forecastData = self.__buildAndForecast__(trainingData)
        return forecastData


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

        for i in range(self.lags):
            trainingData[f'value_lag_diff_{i+1}'] = trainingData[f'value_lag_{i}'] - trainingData[f'value_lag_{i+1}']

        return trainingData

    def __buildAndForecast__(self, trainingData:pd.DataFrame) -> pd.DataFrame:
        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        nextForecastDate = endDate + timedelta(days=1)
        forecastDays = self.ctx['forecastDays']
        train = trainingData[:endDate].copy()
        pred:pd.DataFrame = trainingData[nextForecastDate:].copy()
        train = train.dropna()
        xtraDataPdfTrain = self.xtraDataPdf[self.xtraDataPdf.index.isin(train.index)]
        train_features = train[[f'value_lag_diff_{i+2}' for i in range(self.lags-1)]]
        train_features = pd.concat([train_features, xtraDataPdfTrain], axis=1)
        train_labels = train[['value_lag_diff_1']]
        dtr = DecisionTreeRegressor(max_depth=train_features.shape[1])
        dtr.fit(train_features, train_labels)

        for i in range(forecastDays):
            pred_features = pred.filter(items=[nextForecastDate], axis=0)[[f'value_lag_diff_{i+2}' for i in range(self.lags-1)]]
            xtraDataPdfPred = self.xtraDataPdf[self.xtraDataPdf.index.isin(pred_features.index)]
            pred_features = pd.concat([pred_features, xtraDataPdfPred], axis=1)
            pred_labels = dtr.predict(pred_features)
            nextForecastValueLagDiff1 = pred_labels[0]
            nextForecastValueLag1 = pred['value_lag_1'][nextForecastDate]
            nextForecastValue = nextForecastValueLag1+nextForecastValueLagDiff1
            pred['value_lag_0'][nextForecastDate] = nextForecastValue
            pred['value_lag_diff_1'][nextForecastDate] = nextForecastValueLagDiff1
            thisForecastDate = nextForecastDate
            nextForecastDate = thisForecastDate + timedelta(days=1)
            nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate)
            pred.loc[nextForecastDate] = nextRow


        return pd.DataFrame({"forecastDate": thisForecastDate, "value": nextForecastValue, "forecastModel": self.name}, index=[0])

    def __getNextRow__(self, pred:pd.DataFrame, nextForecastValue:float, thisForecastDate:datetime):
        row = []
        for i in range(self.lags+1):
            if(i == 0):
                row.append(float("nan"))
            elif(i == 1):
                row.append(nextForecastValue)
            else:
                row.append(pred[f'value_lag_{i-1}'][thisForecastDate])

        for i in range(self.lags):
            if(i == 0):
                row.append(float("nan"))
            else:
                row.append(pred[f'value_lag_diff_{i}'][thisForecastDate])

        return row

