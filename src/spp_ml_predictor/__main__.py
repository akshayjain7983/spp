from pyspark.sql import SparkSession
from .dao import SppMLTrainingDao
from .dao import QueryFiles
from .trainer import SppTrainer
import sys
from datetime import datetime, timezone, timedelta

def runSpp(ctx:dict):

    sppMLTrainingDao:SppMLTrainingDao = SppMLTrainingDao.SppMLTrainingDao()
    sppTrainer:SppTrainer = SppTrainer.SppTrainer(sppMLTrainingDao, ctx)
    sppTrainer.train()

def main(args):
    QueryFiles.load()

    pScoreDate = args[0]
    forecastDays = args[1]
    exchange = args[2]
    index = args[3]
    dataHistoryYears = args[4]
    exchangeCode = args[5] if len(args) >= 6 else None

    trainingDataDays = 365*int(dataHistoryYears)
    trainingEndDate = pScoreDate
    trainingStartDate = datetime.strftime(datetime.strptime(trainingEndDate, '%Y-%m-%d') - timedelta(days=trainingDataDays), '%Y-%m-%d')

    for i in range(365):
        ctx = {'exchange': exchange
               , 'pScoreDate':pScoreDate
               , 'trainingStartDate': trainingStartDate
               , 'trainingEndDate': trainingEndDate
               , 'index': index
               , 'exchangeCode': exchangeCode
               , 'forecastDays': int(forecastDays)}

        runSpp(ctx)
        pScoreDate = datetime.strftime(datetime.strptime(pScoreDate, '%Y-%m-%d') + timedelta(days=1), '%Y-%m-%d')
        trainingEndDate = pScoreDate
        trainingStartDate = datetime.strftime(datetime.strptime(trainingEndDate, '%Y-%m-%d') - timedelta(days=trainingDataDays), '%Y-%m-%d')



if __name__ == '__main__':
    main(sys.argv[1:])