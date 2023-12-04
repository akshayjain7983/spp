import pyspark.sql as ps
import pandas as pd
from ..trainer.SppForecastTask import SppForecastTask


class SppIndexForecastTask(SppForecastTask):
    def __init__(self, indexLevelsData:ps.DataFrame, ctx:dict, xtraDataPdf:ps.DataFrame):
        super().__init__(indexLevelsData, ctx, xtraDataPdf)
        self.name = "SppIndexForecastTask"

    def __preForecast__(self) -> pd.DataFrame:
        self.ctx['mode'] = 'index'
        super().__setupCandlestickPatternLags__()
        indexReturnsDataForTraining = self.trainingDataForForecasting.rename(columns={"close": "value"})
        return indexReturnsDataForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:

        self.ctx['mode'] = None
        forecastResult = forecast.copy()
        forecastResult.drop(index=0, inplace=True)
        forecastResult.insert(0, "exchange", trainingDataForForecasting['exchange'][0])
        forecastResult.insert(1, "index", trainingDataForForecasting['index'][0])
        forecastResult.insert(2, "date", self.ctx['pScoreDate'])

        forecastDays = self.ctx['forecastDays']
        indexLevelsPScoreDate = forecast['forecastValues'][0]['value']
        for f in forecastDays[1:]:
            forecastResult.at[f, 'indexReturns'] = forecast['forecastValues'][f]['value'] / indexLevelsPScoreDate - 1

        return forecastResult

