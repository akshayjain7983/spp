import pandas as pd
from ..trainer.SppForecaster import SppForecaster
from ..trainer.SppRandomForests import SppRandomForests
from ..trainer.SppNN import SppNN
from ..trainer.SppLSTM import SppLSTM

class SppForecastTask:
    def __init__(self, trainingDataForForecasting:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.trainingDataForForecasting = trainingDataForForecasting
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def __invokeForecastor__(self, trainingData:pd.DataFrame) ->pd.DataFrame:
        forecastor = self.ctx['forecastor']
        sppForecaster:SppForecaster = None
        match forecastor:
            case "SppRandomForests":
                sppForecaster = SppRandomForests(trainingData, self.ctx, self.xtraDataPdf)
            case "SppNN":
                sppForecaster = SppNN(trainingData, self.ctx, self.xtraDataPdf)
            case "SppLSTM":
                sppForecaster = SppLSTM(trainingData, self.ctx, self.xtraDataPdf)


        return sppForecaster.forecast()

    def __preForecast__(self) -> pd.DataFrame:
        return self.trainingDataForForecasting

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:
        return forecast

    def forecast(self) -> pd.DataFrame:
        trainingData = self.__preForecast__()
        forecast = self.__invokeForecastor__(trainingData)
        finalForecastResult = self.__postForecast__(trainingData, forecast)
        return finalForecastResult