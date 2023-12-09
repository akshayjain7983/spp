import pyspark.sql as ps
import pyspark.sql.functions as psf
import pandas as pd
from ..trainer.SppForecaster import SppForecaster
from ..trainer.SppRandomForests import SppRandomForests
from ..util.SppUtil import ffill
from datetime import datetime, timedelta
from pyspark.sql.types import DoubleType
# from ..trainer.SppNN import SppNN

class SppForecastTask:
    def __init__(self, trainingDataForForecasting:ps.DataFrame, ctx:dict, xtraDataPdf:ps.DataFrame):
        self.trainingDataForForecasting = trainingDataForForecasting
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def __invokeForecastor__(self, trainingData:ps.DataFrame) ->pd.DataFrame:
        forecastor = self.ctx['forecastor']
        sppForecaster:SppForecaster = None
        match forecastor:
            case "SppRandomForests":
                sppForecaster = SppRandomForests(trainingData, self.ctx, self.xtraDataPdf)
            # case "SppNN":
            #     sppForecaster = SppNN(trainingData, self.ctx, self.xtraDataPdf)


        return sppForecaster.forecast()

    def __preForecast__(self) -> ps.DataFrame:
        return self.trainingDataForForecasting

    def __postForecast__(self, trainingDataForForecasting:ps.DataFrame, forecast:ps.DataFrame) -> pd.DataFrame:
        return forecast

    def forecast(self) -> ps.DataFrame:
        trainingData = self.__preForecast__()
        forecast = self.__invokeForecastor__(trainingData)
        finalForecastResult = self.__postForecast__(trainingData, forecast)
        return finalForecastResult

    def __setupCandlestickPatternLags__(self):

        self.xtraDataPdf = self.xtraDataPdf.withColumn('tempPartitioning', psf.lit('tempPartitioning'))  
        lagWindow = ps.Window.partitionBy('tempPartitioning').orderBy('date')
        candleStickLags = 10
        #generate candlestick lags
        candleStickLagCols = [psf.col('*')]
        for i in range(1, candleStickLags):
            lagCol = psf.lag('candlestickMovementReal', i).over(lagWindow).name(f'candlestickMovementRealLag{i}').cast(DoubleType())
            candleStickLagCols.append(lagCol)
        
        self.xtraDataPdf = self.xtraDataPdf.select(candleStickLagCols).drop('tempPartitioning')
        
        #carry forward candlestick lags
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')        
        fillRangeColEnd = endDate + timedelta(days=candleStickLags)
        fillRangeColStart = endDate + timedelta(days=1)
        self.xtraDataPdf = ffill(self.xtraDataPdf, 'candlestickMovementReal', None, 'date', 'date', fillRangeColStart, fillRangeColEnd)
        self.xtraDataPdf = self.xtraDataPdf.fillna(0.0, 'candlestickMovementReal')
        for i in range(1, candleStickLags):
            fillRangeColStart = endDate + timedelta(days=i+1)
            self.xtraDataPdf = ffill(self.xtraDataPdf, f'candlestickMovementRealLag{i}', None, 'date', 'date', fillRangeColStart, fillRangeColEnd)
            self.xtraDataPdf = self.xtraDataPdf.fillna(0.0, f'candlestickMovementRealLag{i}')