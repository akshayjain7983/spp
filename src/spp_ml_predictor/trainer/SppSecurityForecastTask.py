
from datetime import datetime, timezone

import pandas as pd
import time
from ..trainer import SppArima

class SppSecurityForecastTask:
    def __init__(self, forecastIndexReturns:pd.DataFrame, securityDataForExchangeCode:pd.DataFrame, ctx:dict):
        self.forecastIndexReturns = forecastIndexReturns
        self.securityDataForExchangeCode = securityDataForExchangeCode
        self.ctx = ctx

    def buildModel(self) -> pd.DataFrame:

        startT = time.time();
        securityDataForExchangeCodeForTraining = self.securityDataForExchangeCode.rename(columns={"securityReturns90D": "value"})
        forecast = SppArima.buildModel(securityDataForExchangeCodeForTraining, self.ctx)

        forecastedSecurityReturn = forecast['value'][0]
        forecastedIndexReturn = self.forecastIndexReturns['forecast90DIndexReturns'][0]
        forecastedPScore = (forecastedSecurityReturn - forecastedIndexReturn) * 100

        forecast.drop("value", axis=1, inplace=True)
        forecast.insert(0, "exchange", securityDataForExchangeCodeForTraining['exchange'][0])
        forecast.insert(1, "index", self.forecastIndexReturns['index'][0])
        forecast.insert(2, "exchangeCode", securityDataForExchangeCodeForTraining['exchangeCode'][0])
        forecast.insert(3, "isin", securityDataForExchangeCodeForTraining['isin'][0])
        forecast.insert(4, "date", self.ctx['pScoreDate'])
        forecast["forecastedIndexReturn"] = [forecastedIndexReturn]
        forecast["forecastedSecurityReturn"] = [forecastedSecurityReturn]
        forecast["forecastedPScore"] = [forecastedPScore]
        forecast["lastUpdatedTimestamp"] = [datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%dT%H:%M:%S%z')]
        endT = time.time()
        print("SppSecurityForecastTask - Time taken:"+str(endT-startT)+" secs")
        return forecast