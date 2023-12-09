import pyspark.sql as ps
import pyspark.sql.functions as psf
import pandas as pd
from datetime import datetime, date, timedelta
from ..trainer.SppForecaster import SppForecaster
import numpy as np
from pyspark.sql.types import DoubleType

class SppMLForecaster(SppForecaster):
    def __init__(self, trainingDataPdf:ps.DataFrame, ctx:dict, xtraDataPdf:ps.DataFrame):
        super().__init__(trainingDataPdf, ctx, xtraDataPdf)
        self.lags = 60


    def forecast(self) -> pd.DataFrame:
        trainingData = self.__setupTrainingData__()
        forecastData = self.__buildAndForecast__(trainingData)
        return forecastData


    def __setupTrainingData__(self) -> pd.DataFrame:
        
        trainingData:ps.DataFrame = self.trainingDataPdf.select(psf.col('date'), psf.col('value').cast(DoubleType()))
        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        nextForecastDate = endDate + timedelta(days=1)
        trainingData = trainingData[(trainingData.date >= startDate) & (trainingData.date <= nextForecastDate)]
        trainingData = trainingData.orderBy('date').withColumnRenamed('value', 'value_lag_0')
        
        trainingData = trainingData.withColumn('tempPartitioning', psf.lit('tempPartitioning'))  
        lagWindow = ps.Window.partitionBy('tempPartitioning').orderBy('date')
        colsTrainingData = [psf.col('date'), psf.col('tempPartitioning'), psf.col('value_lag_0')]
        
        for i in range(self.lags):
            lagCol = psf.lag('value_lag_0', i+1).over(lagWindow).name(f'value_lag_{i+1}').cast(DoubleType())
            colsTrainingData.append(lagCol)

        for i in range(self.lags+1):
            lagCol = psf.log(f'value_lag_{i}').name(f'value_lag_log_{i}').cast(DoubleType())
            colsTrainingData.append(lagCol)
            
        for i in range(self.lags):
            lagCol =  (psf.col(f'value_lag_log_{i}') - psf.col(f'value_lag_log_{i+1}')).name(f'value_lag_log_diff_{i+1}').cast(DoubleType())
            colsTrainingData.append(lagCol)
            
        trainingData = trainingData.select(colsTrainingData).drop('tempPartitioning')

        return trainingData

    def __getRegressor__(self, train_features:ps.DataFrame, train_labels:ps.DataFrame):
        return None

    def __preparePredFeatures__(self, pred_features:ps.DataFrame):
        return pred_features

    def __predict__(self, regressor, pred_features:ps.DataFrame):
        pred_features = pred_features.dropna();
        return regressor.transform(pred_features)

    def __buildAndForecast__(self, trainingData:ps.DataFrame) -> ps.DataFrame:

        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        endDate = endDate - timedelta(days=1)
        nextForecastDate = endDate + timedelta(days=1)
        forecastDays = self.ctx['forecastDays']
        
        train = trainingData.filter(trainingData.date <= endDate).dropna()
        pred:ps.DataFrame = trainingData.filter(trainingData.date >= nextForecastDate)
        xtraDataPdfTrain = self.xtraDataPdf.join(train, 'date', 'inner').select(self.xtraDataPdf.columns)
        featureCols = [f'value_lag_log_diff_{i+2}' for i in range(self.lags-1)]
        featureCols.insert(0, 'date')
        
        train_features = train.select(featureCols)
        train_features = train_features.join(xtraDataPdfTrain, 'date', 'inner')
        train_labels = train.select('date', 'value_lag_log_diff_1')
        
        regressor = self.__getRegressor__(train_features, train_labels)
        
        for i in range(forecastDays[-1]+1):
            pred_features = pred.filter(pred.date == nextForecastDate).select(featureCols)
            pred_features = pred_features.join(self.xtraDataPdf, 'date', 'inner')
            pred_features = self.__preparePredFeatures__(pred_features)
            pred_labels:ps.DataFrame = self.__predict__(regressor, pred_features)
            nextForecastValueLagLogDiff1 = pred_labels.first()['prediction']
            nextForecastValueLagLog1 = pred.filter(pred.date == nextForecastDate).first()['value_lag_log_1']
            nextForecastValueLagLog = nextForecastValueLagLog1 + nextForecastValueLagLogDiff1
            nextForecastValue = np.exp(nextForecastValueLagLog)
            pred = pred.select([psf.when(pred.date == nextForecastDate, nextForecastValue).otherwise(pred.value_lag_0).name(c[0]) if(c[0] == 'value_lag_0') else psf.col(c[0]) for c in pred.dtypes])
            pred = pred.select([psf.when(pred.date == nextForecastDate, nextForecastValueLagLog).otherwise(pred.value_lag_log_0).name(c[0]) if(c[0] == 'value_lag_log_0') else psf.col(c[0]) for c in pred.dtypes])
            pred = pred.select([psf.when(pred.date == nextForecastDate, nextForecastValueLagLogDiff1).otherwise(pred.value_lag_log_diff_1).name(c[0]) if(c[0] == 'value_lag_log_diff_1') else psf.col(c[0]) for c in pred.dtypes])
            thisForecastDate = nextForecastDate
            nextForecastDate = thisForecastDate + timedelta(days=1)
            pred = self.__addNextRow__(pred, nextForecastValue, thisForecastDate, nextForecastDate)
            pred.show()
        
        return None

        # for i in range(forecastDays[-1]+1):
        #     pred_features = pred.filter(items=[nextForecastDate], axis=0)[[f'value_lag_log_diff_{i+2}' for i in range(self.lags-1)]]
        #     xtraDataPdfPred = self.xtraDataPdf[self.xtraDataPdf.index.isin(pred_features.index)]
        #     pred_features = pd.concat([pred_features, xtraDataPdfPred], axis=1)
        #     pred_features = self.__preparePredFeatures__(pred_features)
        #     pred_labels = self.__predict__(regressor, pred_features)
        #     nextForecastValueLagLogDiff1 = pred_labels[0]
        #     nextForecastValueLagLog1 = pred['value_lag_log_1'][nextForecastDate]
        #     nextForecastValueLagLog = nextForecastValueLagLog1 + nextForecastValueLagLogDiff1
        #     nextForecastValue = np.exp(nextForecastValueLagLog)
        #     pred['value_lag_0'][nextForecastDate] = nextForecastValue
        #     pred['value_lag_log_0'][nextForecastDate] = nextForecastValueLagLog
        #     pred['value_lag_log_diff_1'][nextForecastDate] = nextForecastValueLagLogDiff1
        #     thisForecastDate = nextForecastDate
        #     nextForecastDate = thisForecastDate + timedelta(days=1)
        #     nextRow = self.__getNextRow__(pred, nextForecastValue, thisForecastDate)
        #     pred.loc[nextForecastDate] = nextRow
        #
        # forecastValues = {
        #     "forecastModel": self.__getName__(),
        #     "forecastValues": [{"forecastPeriod": str(d) + 'D', "forecastDate": datetime.strftime(pred.index[d], '%Y-%m-%d'), "value": pred['value_lag_0'][d]} for d in forecastDays]
        # }
        #
        # return pd.DataFrame(forecastValues, index=forecastDays)

    def __addNextRow__(self, pred:ps.DataFrame, nextForecastValue:float, thisForecastDate:datetime, nextForecastDate:datetime) -> ps.DataFrame:
        
        lastRow = pred.filter(pred.date == thisForecastDate).first()
        row = (nextForecastDate,)
        for i in range(self.lags+1):
            if(i == 0):
                row += (float("nan"),)
            else:
                row += (lastRow[f'value_lag_{i-1}'],)

        for i in range(self.lags+1):
            if(i == 0):
                row += (float("nan"),)
            else:
                row += (lastRow[f'value_lag_log_{i-1}'],)

        for i in range(self.lags):
            if(i == 0):
                row += (float("nan"),)
            else:
                row += (lastRow[f'value_lag_log_diff_{i}'],)

        spark = self.ctx['spark']
        schema = pred.schema
        newDf = spark.createDataFrame([row], schema)
        pred = pred.union(newDf)

        return pred