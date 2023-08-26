from ..dao import SppMLTrainingDao
import pandas as pd
from ..trainer import SppSecurityForecastTask
from ..trainer import SppIndexForecastTask
from concurrent.futures import *
from datetime import datetime
import time

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def __submitForSppSecurityForecastTask__(self, forecastIndexReturns:pd.DataFrame, securityReturnsPdf:pd.DataFrame) -> pd.DataFrame:

        securityReturnsPdfLocal = securityReturnsPdf.copy()
        securityReturnsPdfLocal['datetime'] = pd.to_datetime(securityReturnsPdfLocal['date'])
        securityReturnsPdfLocal.set_index("datetime", inplace=True, drop=True)
        securityReturnsPdfLocal.sort_index(inplace=True)
        securityReturnsReindexPdf = pd.date_range(start=datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d'),
                                                  end=datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d'),
                                                  inclusive="both")
        securityReturnsPdfLocal = securityReturnsPdfLocal.reindex(securityReturnsReindexPdf, method='ffill')
        securityReturnsPdfLocal.dropna(inplace=True) #sometimes ffill with reindex may result in nan at begining so drop those
        securityReturns90DSeries = [d.get('90D').get('return') for d in securityReturnsPdfLocal["returns"]]
        securityReturnsPdfLocal.drop("returns", axis=1, inplace=True)
        securityReturnsPdfLocal["securityReturns90D"] = securityReturns90DSeries
        sppTrainingTask = SppSecurityForecastTask.SppSecurityForecastTask(forecastIndexReturns, securityReturnsPdfLocal, self.ctx)
        forecast = sppTrainingTask.buildModel()
        self.sppMLTrainingDao.saveForecastPScore(forecast)
        return forecast;

    def __submitForSppIndexForecastTask__(self, indexReturnsPdf:pd.DataFrame) -> pd.DataFrame:

        indexReturnsPdfLocal = indexReturnsPdf.copy()
        indexReturnsPdfLocal['datetime'] = pd.to_datetime(indexReturnsPdfLocal['date'])
        indexReturnsPdfLocal.set_index("datetime", inplace=True, drop=True)
        indexReturnsPdfLocal.sort_index(inplace=True)
        indexReturnsReindexPdf = pd.date_range(start=datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d'),
                                               end=datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d'),
                                               inclusive="both")
        indexReturnsPdfLocal = indexReturnsPdfLocal.reindex(indexReturnsReindexPdf, method='ffill')
        indexReturnsPdfLocal.dropna(inplace=True) #sometimes ffill with reindex may result in nan at begining so drop those
        indexReturns90DSeries = [d.get('90D').get('return') for d in indexReturnsPdfLocal["returns"]]
        indexReturnsPdfLocal.drop("returns", axis=1, inplace=True)
        indexReturnsPdfLocal["indexReturns90D"] = indexReturns90DSeries
        sppTrainingTask = SppIndexForecastTask.SppIndexForecastTask(indexReturnsPdfLocal, self.ctx)
        forecast = sppTrainingTask.buildModel()
        return forecast;

    def train(self):

        startT = time.time()
        exchangeCodeDf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityExchangeCodes(self.ctx)
        exchangeCodeDf = exchangeCodeDf[["exchangeCode"]].copy()
        securityReturnsPdf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityReturns(exchangeCodeDf['exchangeCode'], self.ctx)
        indexReturnsPdf:pd.DataFrame = self.sppMLTrainingDao.loadIndexReturns(self.ctx)

        forecastIndexReturns = self.__submitForSppIndexForecastTask__(indexReturnsPdf)


        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for ec in exchangeCodeDf['exchangeCode']:
                future = executor.submit(self.__submitForSppSecurityForecastTask__, forecastIndexReturns, securityReturnsPdf[securityReturnsPdf['exchangeCode'] == ec])
                futures.append(future)


        for f in futures:
            print(f.result())

        endT = time.time()
        print("SppTrainer - "+self.ctx['pScoreDate']+" - Time taken:" + str(endT - startT) + " secs")