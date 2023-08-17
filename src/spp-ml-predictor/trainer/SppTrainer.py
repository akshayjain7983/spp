from typing import Iterator

from ..dao import SppMLTrainingDao
import pyspark.sql as ps
from pylab import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from ..trainer import SppTrainingTask
from pyspark.sql.functions import pandas_udf
from pyspark.sql.streaming.state import GroupState
from pyspark.sql.streaming.state import Tuple

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def train(self):
        exchangeCodeDf:ps.DataFrame = self.sppMLTrainingDao.loadSecurityExchangeCodes(self.ctx)
        exchangeCodeDf = exchangeCodeDf.select("exchangeCode").cache()
        exchangeCodeList = exchangeCodeDf.collect()

        trainingPscoreDf:ps.DataFrame = self.sppMLTrainingDao.loadSecurityTrainingPScore(exchangeCodeList, self.ctx)
        trainingPscoreDf = trainingPscoreDf.select("exchangeCode", "date", "trainingPScore.90D.pScore")
        trainingPscoreDf.show()
        def submitForTrainingTask(psdf:pd.DataFrame) -> pd.DataFrame:
            sppTrainingTask = SppTrainingTask.SppTrainingTask(psdf)
            return sppTrainingTask.buildModel()

        # trainingPscoreDf.groupBy(trainingPscoreDf.exchangeCode).applyInPandas(submitForTrainingTask, "date string, pScore double").show()
        forecast = submitForTrainingTask(trainingPscoreDf.toPandas())


        # futures = []
        #
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     for ec in exchangeCodeList:
        #         sppTrainingTask = SppTrainingTask.SppTrainingTask(ec.exchangeCode, trainingPscoreDf, self.ctx)
        #         future = executor.submit(sppTrainingTask.buildModel)
        #         futures.append(future)
        #
        #     for f in concurrent.futures.as_completed(futures):
        #         data = f.result()


