from typing import Iterator

from ..dao import SppMLTrainingDao
import pyspark.sql as ps
from pylab import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from ..trainer import SppTrainingTask
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.streaming.state import GroupState
from pyspark.sql.streaming.state import Tuple

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def train(self):
        print(self.ctx)
        exchangeCodeDf:ps.DataFrame = self.sppMLTrainingDao.loadSecurityExchangeCodes(self.ctx)
        exchangeCodeDf = exchangeCodeDf.select("exchangeCode").cache()
        exchangeCodeList = exchangeCodeDf.collect()
        print(exchangeCodeList)
        trainingPscoreDf:ps.DataFrame = self.sppMLTrainingDao.loadSecurityTrainingPScore(exchangeCodeList, self.ctx)
        trainingPscoreDf.printSchema()
        trainingPscoreDf.show()

        def submitForTrainingTask(psdf:pd.DataFrame) -> pd.DataFrame:
            psdfForTraining = pd.json_normalize(psdf['doc'][0])
            psdfForTraining = psdfForTraining[["exchange", "index", "exchangeCode", "isin", "date", "trainingPScore.90D.pScore"]]
            psdfForTraining.rename(columns={"trainingPScore.90D.pScore":"pScore"}, inplace=True)
            psdfForTraining.set_index("date", inplace=True)
            psdfForTraining.sort_index(inplace=True)
            sppTrainingTask = SppTrainingTask.SppTrainingTask(psdfForTraining)
            forecast = sppTrainingTask.buildModel()
            forecast.insert(0, "exchange", psdfForTraining['exchange'][0])
            forecast.insert(1, "index", psdfForTraining['index'][0])
            forecast.insert(2, "exchangeCode", psdfForTraining['exchangeCode'][0])
            forecast.insert(3, "isin", psdfForTraining['isin'][0])
            forecast.rename(columns={"pScore":"forecastPScore90D"}, inplace=True)
            return forecast;



        sparkMode = self.ctx['sparkMode']

        if(sparkMode == 'submit'):
            forecast = trainingPscoreDf.groupBy('_id').applyInPandas(submitForTrainingTask, 'exchange string, index string, exchangeCode string, isin string, date string, forecastPScore90D double')
            forecast.show()

        else:
            forecast = submitForTrainingTask(trainingPscoreDf.toPandas())
            print(forecast)


