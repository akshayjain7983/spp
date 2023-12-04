import pyspark.sql as ps
import pyspark.sql.functions as psf
import pandas as pd
import time
from ..trainer.SppForecaster import SppForecaster
from ..trainer.SppRandomForests import SppRandomForests
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

        lagWindow = ps.Window.orderBy('date')
        candleStickLags = 9
        for i in range(1, candleStickLags):
            self.xtraDataPdf = self.xtraDataPdf.withColumn(f'candlestickMovementRealLag{i}', psf.lag('candlestickMovementReal', i).over(lagWindow))
        


        candleStickLags = 5
        for i in range(1, candleStickLags):
            self.xtraDataPdf[f'candleStickRealBodyChangeLag{i}'] = self.xtraDataPdf['candleStickRealBodyChange'].shift(i)
            self.xtraDataPdf[f'candleStickRealBodyChangeLag{i}'].ffill(limit=(candleStickLags-i), inplace=True)
            self.xtraDataPdf[f'candleStickRealBodyChangeLag{i}'].replace(to_replace=float('NaN'), value=0.0, inplace=True)
        self.xtraDataPdf['candleStickRealBodyChange'].ffill(limit=candleStickLags, inplace=True)
        self.xtraDataPdf['candleStickRealBodyChange'].replace(to_replace=float('NaN'), value=0.0, inplace=True)