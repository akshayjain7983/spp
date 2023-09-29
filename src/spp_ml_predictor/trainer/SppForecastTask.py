import pandas as pd
import time
from ..trainer.SppArima import SppArima
from ..trainer.SppDecisionTree import SppDecisionTree
from ..trainer.SppForecaster import SppForecaster
from ..trainer.SppRandomForests import SppRandomForests
from ..trainer.SppRidge import SppRidge
from ..trainer.SppPolyRidge import SppPolyRidge
from ..trainer.SppAdaBoost import SppAdaBoost
from ..trainer.SppVotingForecaster import SppVotingForecaster
from ..trainer.SppGradientBoost import SppGradientBoost

class SppForecastTask:
    def __init__(self, trainingDataForForecasting:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.trainingDataForForecasting = trainingDataForForecasting
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def __invokeForecastor__(self, trainingData:pd.DataFrame) ->pd.DataFrame:
        forecastor = self.ctx['forecastor']
        sppForecaster:SppForecaster = None
        match forecastor:
            case "SppArima":
                sppForecaster = SppArima(trainingData, self.ctx, self.xtraDataPdf)
            case "SppDecisionTree":
                sppForecaster = SppDecisionTree(trainingData, self.ctx, self.xtraDataPdf)
            case "SppRandomForests":
                sppForecaster = SppRandomForests(trainingData, self.ctx, self.xtraDataPdf)
            case "SppRidge":
                sppForecaster = SppRidge(trainingData, self.ctx, self.xtraDataPdf)
            case "SppPolyRidge":
                sppForecaster = SppPolyRidge(trainingData, self.ctx, self.xtraDataPdf)
            case "SppAdaBoost":
                sppForecaster = SppAdaBoost(trainingData, self.ctx, self.xtraDataPdf)
            case "SppVotingForecaster":
                sppForecaster = SppVotingForecaster(trainingData, self.ctx, self.xtraDataPdf)
            case "SppGradientBoost":
                sppForecaster = SppGradientBoost(trainingData, self.ctx, self.xtraDataPdf)

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