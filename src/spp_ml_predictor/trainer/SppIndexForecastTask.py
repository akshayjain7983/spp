import pandas as pd
import time
from ..trainer.SppArima import SppArima
from ..trainer.SppDecisionTree import SppDecisionTree
from ..trainer.SppForecaster import SppForecaster


class SppIndexForecastTask:
    def __init__(self, indexReturnsData:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.indexReturnsData = indexReturnsData
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def forecast(self) -> pd.DataFrame:

        startT = time.time();
        indexReturnsDataForTraining = self.indexReturnsData.rename(columns={"indexReturns90D":"value"})
        forecast = self.invokeForecastor(indexReturnsDataForTraining)
        forecast.rename(columns={"value":"forecast"+str(self.ctx['forecastDays'])+"DIndexReturns"}, inplace=True)
        forecast.insert(0, "exchange", indexReturnsDataForTraining['exchange'][0])
        forecast.insert(1, "index", indexReturnsDataForTraining['index'][0])
        forecast.insert(2, "date", self.ctx['pScoreDate'])
        endT = time.time()
        print("SppIndexForecastTask - Time taken:"+str(endT-startT)+" secs")
        return forecast

    def invokeForecastor(self, indexReturnsDataForTraining:pd.DataFrame) ->pd.DataFrame:
        forecastor = self.ctx['forecastor']
        sppForecaster:SppForecaster = None
        match forecastor:
            case "SppArima":
                sppForecaster = SppArima(indexReturnsDataForTraining, self.ctx, self.xtraDataPdf)
            case "SppDecisionTree":
                sppForecaster = SppDecisionTree(indexReturnsDataForTraining, self.ctx, self.xtraDataPdf)

        return sppForecaster.forecast()