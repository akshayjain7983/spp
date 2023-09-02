import pandas as pd
import time
from ..trainer.SppForecastTask import SppForecastTask


class SppIndexForecastTask(SppForecastTask):
    def __init__(self, indexReturnsData:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        super().__init__(indexReturnsData, ctx, xtraDataPdf)
        self.name = "SppIndexForecastTask"

    def __preForecast__(self) -> pd.DataFrame:
        indexReturnsDataForTraining = self.trainingDataForForecasting.rename(columns={"indexReturns90D": "value"})
        return indexReturnsDataForTraining

    def __postForecast__(self, trainingDataForForecasting:pd.DataFrame, forecast:pd.DataFrame) -> pd.DataFrame:
        forecast.rename(columns={"value": "forecast" + str(self.ctx['forecastDays']) + "DIndexReturns"}, inplace=True)
        forecast.insert(0, "exchange", trainingDataForForecasting['exchange'][0])
        forecast.insert(1, "index", trainingDataForForecasting['index'][0])
        forecast.insert(2, "date", self.ctx['pScoreDate'])
        return forecast

    # def forecast(self) -> pd.DataFrame:
    #
    #     startT = time.time();
    #     indexReturnsDataForTraining = self.indexReturnsData.rename(columns={"indexReturns90D":"value"})
    #     forecast = super().__invokeForecastor__(indexReturnsDataForTraining)
    #     forecast.rename(columns={"value":"forecast"+str(self.ctx['forecastDays'])+"DIndexReturns"}, inplace=True)
    #     forecast.insert(0, "exchange", indexReturnsDataForTraining['exchange'][0])
    #     forecast.insert(1, "index", indexReturnsDataForTraining['index'][0])
    #     forecast.insert(2, "date", self.ctx['pScoreDate'])
    #     endT = time.time()
    #     print("SppIndexForecastTask - Time taken:"+str(endT-startT)+" secs")
    #     return forecast

