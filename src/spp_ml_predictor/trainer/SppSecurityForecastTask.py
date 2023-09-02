
from datetime import datetime, timezone

import pandas as pd
import time
from ..trainer.SppForecastTask import SppForecastTask

class SppSecurityForecastTask(SppForecastTask):
    def __init__(self, forecastIndexReturns:pd.DataFrame, securityDataForExchangeCode:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(securityDataForExchangeCode, ctx, xtraDataPdf)
        self.forecastIndexReturns = forecastIndexReturns
        self.name = "SppSecurityForecastTask"


    def __preForecast__(self) -> pd.DataFrame:
        securityDataForExchangeCodeForTraining = self.trainingDataForForecasting.rename(columns={"securityReturns90D": "value"})
        return securityDataForExchangeCodeForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:
        forecastedSecurityReturn = forecast['value'][0]
        forecastedIndexReturn = self.forecastIndexReturns["forecast" + str(self.ctx['forecastDays']) + "DIndexReturns"][0]
        forecastedPScore = (forecastedSecurityReturn - forecastedIndexReturn) * 100

        forecast.drop("value", axis=1, inplace=True)
        forecast.insert(0, "exchange", trainingDataForForecasting['exchange'][0])
        forecast.insert(1, "index", self.forecastIndexReturns['index'][0])
        forecast.insert(2, "exchangeCode", trainingDataForForecasting['exchangeCode'][0])
        forecast.insert(3, "isin", trainingDataForForecasting['isin'][0])
        forecast.insert(4, "date", self.ctx['pScoreDate'])
        forecast["forecastedIndexReturn"] = [forecastedIndexReturn]
        forecast["forecastedSecurityReturn"] = [forecastedSecurityReturn]
        forecast["forecastPeriod"] = [str(self.ctx['forecastDays']) + "D"]
        forecast["forecastedPScore"] = [forecastedPScore]
        forecast["lastUpdatedTimestamp"] = [datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%dT%H:%M:%S%z')]
        return forecast

    # def forecast(self) -> pd.DataFrame:
    #
    #     startT = time.time();
    #     securityDataForExchangeCodeForTraining = self.securityDataForExchangeCode.rename(columns={"securityReturns90D": "value"})
    #     forecast = super().__invokeForecastor__(securityDataForExchangeCodeForTraining)
    #
    #     forecastedSecurityReturn = forecast['value'][0]
    #     forecastedIndexReturn = self.forecastIndexReturns["forecast"+str(self.ctx['forecastDays'])+"DIndexReturns"][0]
    #     forecastedPScore = (forecastedSecurityReturn - forecastedIndexReturn) * 100
    #
    #     forecast.drop("value", axis=1, inplace=True)
    #     forecast.insert(0, "exchange", securityDataForExchangeCodeForTraining['exchange'][0])
    #     forecast.insert(1, "index", self.forecastIndexReturns['index'][0])
    #     forecast.insert(2, "exchangeCode", securityDataForExchangeCodeForTraining['exchangeCode'][0])
    #     forecast.insert(3, "isin", securityDataForExchangeCodeForTraining['isin'][0])
    #     forecast.insert(4, "date", self.ctx['pScoreDate'])
    #     forecast["forecastedIndexReturn"] = [forecastedIndexReturn]
    #     forecast["forecastedSecurityReturn"] = [forecastedSecurityReturn]
    #     forecast["forecastPeriod"] = [str(self.ctx['forecastDays'])+"D"]
    #     forecast["forecastedPScore"] = [forecastedPScore]
    #     forecast["lastUpdatedTimestamp"] = [datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%dT%H:%M:%S%z')]
    #     endT = time.time()
    #     print("SppSecurityForecastTask - Time taken:"+str(endT-startT)+" secs")
    #     return forecast