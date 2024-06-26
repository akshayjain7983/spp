import pandas as pd
from datetime import datetime
from ..trainer.SppForecastTask import SppForecastTask


class SppIndexForecastTask(SppForecastTask):
    def __init__(self, indexLevelsData:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(indexLevelsData, ctx, xtraDataPdf)
        self.name = "SppIndexForecastTask"

    def __preForecast__(self) -> pd.DataFrame:
        self.ctx['mode'] = 'index'
        indexReturnsDataForTraining = self.trainingDataForForecasting.rename(columns={"close": "value"})
        return indexReturnsDataForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:

        self.ctx['mode'] = None
        forecastDays = self.ctx['forecastDays']

        forecastResult = forecast.copy()
        forecastResult.drop(index=0, inplace=True)
        forecastResult.insert(0, "index_id", trainingDataForForecasting['index_id'].iloc[0])
        forecastResult.insert(1, "date", self.ctx['pScoreDate'])
        forecastResult.rename(columns={"forecastModel": "forecast_model_name"}, inplace=True)
        forecastResult.insert(loc=len(forecastResult.columns), column="forecast_period", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecast_date", value="")
        forecastResult.insert(loc=len(forecastResult.columns), column="forecasted_level", value=float('nan'))
        forecastResult.insert(loc=len(forecastResult.columns), column="forecasted_return", value=float('nan'))
        forecastResult.at[forecastDays, "forecast_period"] = forecast['forecastValues'].loc[forecastDays]['forecastPeriod']
        forecastResult.at[forecastDays, "forecast_date"] = forecast['forecastValues'].loc[forecastDays]['forecastDate']

        indexLevelsPScoreDate = forecast['forecastValues'].loc[0]['value']
        forecastResult.at[forecastDays, 'forecasted_level'] = forecast['forecastValues'].loc[forecastDays]['value']
        forecastResult.at[forecastDays, 'forecasted_return'] = forecast['forecastValues'].loc[forecastDays]['value'] / indexLevelsPScoreDate - 1
        return forecastResult

