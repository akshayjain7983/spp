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
        forecastResult.drop(index=(forecastDays*2), inplace=True)
        forecastResult.insert(0, "index_id", trainingDataForForecasting['index_id'].iloc[0])
        forecastResult.insert(1, "date", self.ctx['pScoreDate'])

        indexLevelsPScoreDate = forecast['forecastValues'].loc[forecastDays]['value']
        forecastResult.at[forecastDays, 'indexReturns'] = forecast['forecastValues'].loc[forecastDays*2]['value'] / indexLevelsPScoreDate - 1
        return forecastResult

