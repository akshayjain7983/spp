import pandas as pd
from datetime import datetime
from ..trainer.SppForecastTask import SppForecastTask


class SppIndexForecastTask(SppForecastTask):
    def __init__(self, indexLevelsData:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(indexLevelsData, ctx, xtraDataPdf)
        self.name = "SppIndexForecastTask"

    def __preForecast__(self) -> pd.DataFrame:
        self.ctx['mode'] = 'index'
        super().__setupCandlestickPatternLags__()
        indexReturnsDataForTraining = self.trainingDataForForecasting.rename(columns={"close": "value"})
        return indexReturnsDataForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:

        self.ctx['mode'] = None
        forecast.insert(0, "exchange", trainingDataForForecasting['exchange'][0])
        forecast.insert(1, "index", trainingDataForForecasting['index'][0])
        forecast.insert(2, "date", self.ctx['pScoreDate'])

        forecastDays = self.ctx['forecastDays']
        pScoreDate = datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d')
        indexLevelsPScoreDate = self.trainingDataForForecasting['close'][pScoreDate]
        for f in forecastDays:
            forecast.at[f, 'indexReturns'] = forecast['forecastValues'][f]['value'] / indexLevelsPScoreDate - 1

        return forecast

