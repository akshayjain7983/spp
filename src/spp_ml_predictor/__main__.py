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
    forecastDays = [int(e) for e in args[1].split(',')]
    forecastDays.sort()
    exchange = args[2]
    index = args[3]
    dataHistoryMonths = args[4]
    forecastor = args[5]
    exchangeCode = args[6] if len(args) >= 7 else None

    trainingDataDays = 30*int(dataHistoryMonths)
    trainingEndDate = pScoreDate
    trainingStartDate = datetime.strftime(datetime.strptime(trainingEndDate, '%Y-%m-%d') - timedelta(days=trainingDataDays), '%Y-%m-%d')

    for i in range(365):
        ctx = {'exchange': exchange
               , 'pScoreDate':pScoreDate
               , 'trainingStartDate': trainingStartDate
               , 'trainingEndDate': trainingEndDate
               , 'index': index
               , 'exchangeCode': exchangeCode
               , 'forecastDays': forecastDays
               , 'forecastor': forecastor}

        runSpp(ctx)
        pScoreDate = datetime.strftime(datetime.strptime(pScoreDate, '%Y-%m-%d') + timedelta(days=1), '%Y-%m-%d')
        trainingEndDate = pScoreDate
        trainingStartDate = datetime.strftime(datetime.strptime(trainingEndDate, '%Y-%m-%d') - timedelta(days=trainingDataDays), '%Y-%m-%d')



if __name__ == '__main__':
    main(sys.argv[1:])