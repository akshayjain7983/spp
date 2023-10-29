
from datetime import datetime, timezone, timedelta

import pandas as pd
import time
from ..trainer.SppForecastTask import SppForecastTask

class SppSecurityForecastTask(SppForecastTask):
    def __init__(self, forecastIndexReturns:pd.DataFrame, securityDataForExchangeCode:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(securityDataForExchangeCode, ctx, xtraDataPdf)
        self.forecastIndexReturns = forecastIndexReturns
        self.name = "SppSecurityForecastTask"

    def __preForecast__(self) -> pd.DataFrame:
        super().__setupCandlestickPatternLags__()
        securityDataForExchangeCodeForTraining = self.trainingDataForForecasting.rename(columns={"close": "value"})
        return securityDataForExchangeCodeForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:

        forecastDays = self.ctx['forecastDays']
        securityPricePScoreDate = forecast['forecastValues'][0]['value']
        forecastResult = forecast.copy()
        forecastResult.drop(index=0, inplace=True)
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastPeriod", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastDate", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastedIndexReturn", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastedSecurityReturn", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastedPScore", value=float('nan'))
        forecastResult.drop("forecastValues", axis=1, inplace=True)

        for d in forecastDays[1:]:
            forecastedSecurityPrice = forecast['forecastValues'][d]['value']
            forecastedSecurityReturn = forecastedSecurityPrice / securityPricePScoreDate - 1
            forecastedIndexReturn = self.forecastIndexReturns['indexReturns'][d]
            forecastedPScore = (forecastedSecurityReturn - forecastedIndexReturn) * 100
            forecastResult.at[d, "forecastedIndexReturn"] = forecastedIndexReturn
            forecastResult.at[d, "forecastedSecurityReturn"] = forecastedSecurityReturn
            forecastResult.at[d, "forecastPeriod"] = forecast['forecastValues'][d]['forecastPeriod']
            forecastResult.at[d, "forecastedPScore"] = forecastedPScore
            forecastResult.at[d, "forecastDate"] = forecast['forecastValues'][d]['forecastDate']

        forecastResult.insert(0, "exchange", trainingDataForForecasting['exchange'][0])
        forecastResult.insert(1, "index", self.forecastIndexReturns['index'][forecastDays[1]])
        forecastResult.insert(2, "exchangeCode", trainingDataForForecasting['exchangeCode'][0])
        forecastResult.insert(3, "isin", trainingDataForForecasting['isin'][0])
        forecastResult.insert(4, "date", self.ctx['pScoreDate'])
        forecastResult.insert(5, "lastUpdatedTimestamp", datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%dT%H:%M:%S%z'))
        return forecastResult