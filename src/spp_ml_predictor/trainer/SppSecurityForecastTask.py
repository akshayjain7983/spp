
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
        self.ctx['mode'] = 'security'
        super().__setupCandlestickPatternLags__()
        securityDataForExchangeCodeForTraining = self.trainingDataForForecasting.rename(columns={"close": "value"})
        return securityDataForExchangeCodeForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:
        self.ctx['mode'] = None
        forecastDays = self.ctx['forecastDays']
        securityPricePScoreDate = forecast['forecastValues'][0]['value']
        forecastResult = forecast.copy()
        forecastResult.drop(index=0, inplace=True)
        forecastResult.insert(loc=len(forecastResult.columns), column="forecast_period", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecast_date", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecasted_index_return", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecasted_security_return", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecasted_p_score", value=float('nan'))
        forecastResult.rename(columns={"forecastModel": "forecast_model_name"}, inplace=True)
        forecastResult.drop("forecastValues", axis=1, inplace=True)

        for d in forecastDays[1:]:
            forecastedSecurityPrice = forecast['forecastValues'][d]['value']
            forecastedSecurityReturn = forecastedSecurityPrice / securityPricePScoreDate - 1
            forecastedIndexReturn = self.forecastIndexReturns['indexReturns'][d]
            forecastedPScore = (forecastedSecurityReturn - forecastedIndexReturn) * 100
            forecastResult.at[d, "forecasted_index_return"] = forecastedIndexReturn
            forecastResult.at[d, "forecasted_security_return"] = forecastedSecurityReturn
            forecastResult.at[d, "forecast_period"] = forecast['forecastValues'][d]['forecastPeriod']
            forecastResult.at[d, "forecasted_p_score"] = forecastedPScore
            forecastResult.at[d, "forecast_date"] = forecast['forecastValues'][d]['forecastDate']

        forecastResult.insert(0, "index_id", self.forecastIndexReturns['index_id'][forecastDays[1]])
        forecastResult.insert(2, "security_id", trainingDataForForecasting['security_id'].iloc[0])
        forecastResult.insert(3, "date", self.ctx['pScoreDate'])
        return forecastResult