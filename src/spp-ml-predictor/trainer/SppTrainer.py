from ..dao import SppMLTrainingDao
from pyspark.sql import DataFrame

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def train(self):
        exchangeCodeDf:DataFrame = self.sppMLTrainingDao.loadSecurityExchangeCodes(self.ctx)
        exchangeCodeDf.printSchema()
        exchangeCodeDf.show()

        exchangeCodeDf = exchangeCodeDf.select("exchangeCode").cache()
        exchangeCodeList = exchangeCodeDf.collect()

        trainingPscoreDf:DataFrame = self.sppMLTrainingDao.loadSecurityTrainingPScore(exchangeCodeList, self.ctx)
        trainingPscoreDf.printSchema()
        trainingPscoreDf.show()

