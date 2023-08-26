import pandas as pd
import time
from ..trainer import SppArima


class SppIndexForecastTask:
    def __init__(self, indexReturnsData:pd.DataFrame, ctx:dict):
        self.indexReturnsData = indexReturnsData
        self.ctx = ctx

    def buildModel(self) -> pd.DataFrame:

        startT = time.time();
        indexReturnsDataForTraining = self.indexReturnsData.rename(columns={"indexReturns90D":"value"})
        forecast = SppArima.buildModel(indexReturnsDataForTraining, self.ctx)
        forecast.rename(columns={"value":"forecast90DIndexReturns"}, inplace=True)
        forecast.insert(0, "exchange", indexReturnsDataForTraining['exchange'][0])
        forecast.insert(1, "index", indexReturnsDataForTraining['index'][0])
        forecast.insert(2, "date", self.ctx['pScoreDate'])
        endT = time.time()
        print("SppIndexForecastTask - Time taken:"+str(endT-startT)+" secs")
        return forecast