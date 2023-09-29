
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
        self.__setupCandlestickPatternLags__()
        securityDataForExchangeCodeForTraining = self.trainingDataForForecasting.rename(columns={"close": "value"})
        return securityDataForExchangeCodeForTraining

    def __setupCandlestickPatternLags__(self):

        candleStickLags = 5
        for i in range(1, candleStickLags):
            self.xtraDataPdf[f'candleStickRealBodyChangeLag{i}'] = self.xtraDataPdf['candleStickRealBodyChange'].shift(i)
            self.xtraDataPdf[f'candleStickRealBodyChangeLag{i}'].ffill(limit=(candleStickLags-i), inplace=True)
            self.xtraDataPdf[f'candleStickRealBodyChangeLag{i}'].replace(to_replace=float('NaN'), value=0.0, inplace=True)
        self.xtraDataPdf['candleStickRealBodyChange'].ffill(limit=candleStickLags, inplace=True)
        self.xtraDataPdf['candleStickRealBodyChange'].replace(to_replace=float('NaN'), value=0.0, inplace=True)

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:

        forecastDays = self.ctx['forecastDays']
        pScoreDate = datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d')
        securityPricePScoreDate = self.trainingDataForForecasting['close'][pScoreDate]
        forecastResult = forecast.copy()
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastPeriod", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastDate", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastedIndexReturn", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastedSecurityReturn", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecastedPScore", value=float('nan'))
        forecastResult.drop("forecastValues", axis=1, inplace=True)

        for d in forecastDays:
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
        forecastResult.insert(1, "index", self.forecastIndexReturns['index'][forecastDays[0]])
        forecastResult.insert(2, "exchangeCode", trainingDataForForecasting['exchangeCode'][0])
        forecastResult.insert(3, "isin", trainingDataForForecasting['isin'][0])
        forecastResult.insert(4, "date", self.ctx['pScoreDate'])
        forecastResult.insert(5, "lastUpdatedTimestamp", datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%dT%H:%M:%S%z'))
        return forecastResult