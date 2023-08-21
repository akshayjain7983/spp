from typing import Iterator

from ..dao import SppMLTrainingDao
# import pyspark.sql as ps
# import pyspark.sql.functions as psf
from pylab import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from ..trainer import SppTrainingTask
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.streaming.state import GroupState
from pyspark.sql.streaming.state import Tuple
from datetime import datetime,timezone
from concurrent.futures import *

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def __submitForTrainingTask__(self, psdf: pd.DataFrame) -> pd.DataFrame:
        print("====================starting submitForTrainingTask==========")
        print(psdf)
        psdfForTraining = pd.json_normalize(psdf['doc'][0])
        psdfForTraining = psdfForTraining[
            ["exchange", "index", "exchangeCode", "isin", "date", "trainingPScore.90D.pScore"]]
        psdfForTraining.rename(columns={"trainingPScore.90D.pScore": "pScore"}, inplace=True)
        psdfForTraining.set_index("date", inplace=True)
        psdfForTraining.sort_index(inplace=True)
        sppTrainingTask = SppTrainingTask.SppTrainingTask(psdfForTraining)
        forecast = sppTrainingTask.buildModel()
        forecast.insert(0, "exchange", psdfForTraining['exchange'][0])
        forecast.insert(1, "index", psdfForTraining['index'][0])
        forecast.insert(2, "exchangeCode", psdfForTraining['exchangeCode'][0])
        forecast.insert(3, "isin", psdfForTraining['isin'][0])
        print("====================finishing submitForTrainingTask==========")
        return forecast;

    def train(self):
        print(self.ctx)
        exchangeCodeDf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityExchangeCodes(self.ctx)
        exchangeCodeDf = exchangeCodeDf[["exchangeCode"]].copy()
        # exchangeCodeDf = exchangeCodeDf.select("exchangeCode").cache()
        # exchangeCodeList = exchangeCodeDf.collect()
        print(exchangeCodeDf)
        trainingPscoreDf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityTrainingPScore(exchangeCodeDf['exchangeCode'], self.ctx)
        print(trainingPscoreDf)

        futures = []

        with ThreadPoolExecutor(max_workers=1) as executor:

            for ec in trainingPscoreDf['_id']:
                future = executor.submit(self.__submitForTrainingTask__, trainingPscoreDf[trainingPscoreDf['_id'] == ec])
                futures.append(future)


        for f in futures:
            print(f.result())



        # sparkMode = self.ctx['sparkMode']

        # if(sparkMode == 'submit'):
        #     forecast = trainingPscoreDf.repartitionByRange(1, "_id").groupBy('_id').applyInPandas(submitForTrainingTask, 'exchange string, index string, exchangeCode string, isin string, date string'
        #                                                                                     ', forecastPScore double, forecastPScoreModel string')
        #
        #     forecast = (forecast
        #                 .withColumnRenamed("forecastPScore", "forecastPScore.90D.pScore")
        #                 .withColumnRenamed("forecastPScoreModel", "forecastPScore.90D.model")
        #                 .withColumn("lastUpdatedTimestamp", psf.lit(datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%dT%H:%M:%S%z'))))
        #     forecast.show()
        #     self.sppMLTrainingDao.saveForecastPScore(forecast)
        #
        # else:
        #     forecast = submitForTrainingTask(trainingPscoreDf.toPandas())
        #     print(forecast)

