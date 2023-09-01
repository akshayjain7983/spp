import pandas as pd
import time
from ..trainer import SppArima
from ..trainer import SppDecisionTree


class SppIndexForecastTask:
    def __init__(self, indexReturnsData:pd.DataFrame, ctx:dict, xtraDataPdf:pd.DataFrame):
        self.indexReturnsData = indexReturnsData
        self.ctx = ctx
        self.xtraDataPdf = xtraDataPdf

    def buildModel(self) -> pd.DataFrame:

        startT = time.time();
        indexReturnsDataForTraining = self.indexReturnsData.rename(columns={"indexReturns90D":"value"})
        forecast = self.invokeRegressor(indexReturnsDataForTraining)
        forecast.rename(columns={"value":"forecast"+str(self.ctx['forecastDays'])+"DIndexReturns"}, inplace=True)
        forecast.insert(0, "exchange", indexReturnsDataForTraining['exchange'][0])
        forecast.insert(1, "index", indexReturnsDataForTraining['index'][0])
        forecast.insert(2, "date", self.ctx['pScoreDate'])
        endT = time.time()
        print("SppIndexForecastTask - Time taken:"+str(endT-startT)+" secs")
        return forecast

    def invokeRegressor(self, indexReturnsDataForTraining:pd.DataFrame) ->pd.DataFrame:
        regressor = self.ctx['regressor']
        match regressor:
            case "SppArima":
                sppRegressor = SppArima.SppArima(indexReturnsDataForTraining, self.ctx, self.xtraDataPdf)
                return sppRegressor.forecast()
            case "SppDecisionTree":
                sppRegressor = SppDecisionTree.SppDecisionTree(indexReturnsDataForTraining, self.ctx, self.xtraDataPdf)
                return sppRegressor.forecast()
            case default: return None