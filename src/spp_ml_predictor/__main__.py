import pyspark.sql as ps
from .dao import SppMLTrainingDao
from .dao import QueryFiles
from .trainer import SppTrainer
from .trainer import SppTrainingDataTransformer
import sys
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY

def runSpp(ctx:dict, sppMLTrainingDao: SppMLTrainingDao, spark: ps.SparkSession):

    sppTrainer:SppTrainer = SppTrainer.SppTrainer(sppMLTrainingDao, ctx, spark)
    sppTrainer.train()


def extractTrainingData(trainingData:dict, trainingStartDate, trainingEndDate):

    indexLevelsPdf:ps.DataFrame = trainingData['indexLevelsPdf']
    indexLevelsPdfForTraining = indexLevelsPdf[(indexLevelsPdf.date >= trainingStartDate) & (indexLevelsPdf.date <= trainingEndDate)]
    securityPricesPdf:ps.DataFrame = trainingData['securityPricesPdf']
    securityPricesPdfForTraining = securityPricesPdf[(securityPricesPdf.tradingDate >= trainingStartDate) & (securityPricesPdf.tradingDate <= trainingEndDate)]
    trainingDataCopy = trainingData.copy()
    trainingDataCopy['indexLevelsPdf'] = indexLevelsPdfForTraining
    trainingDataCopy['securityPricesPdf'] = securityPricesPdfForTraining
    return trainingDataCopy

def main(args):
    QueryFiles.load()

    pScoreStartDate = args[0]
    pScoreEndDate = args[1]
    forecastDays = [int(e) for e in args[2].split(',')]
    forecastDays.append(0)
    forecastDays.sort()
    exchange = args[3]
    index = args[4]
    dataHistoryMonths = args[5]
    forecastor = args[6]
    exchangeCode = args[7] if len(args) >= 8 else None

    trainingDataDays = 31*int(dataHistoryMonths)
    
    spark = ps.SparkSession \
                    .builder \
                    .appName("spp") \
                    .config("spark.mongodb.input.uri", "mongodb://localhost:27017") \
                    .config("spark.mongodb.output.uri", "mongodb://localhost:27017") \
                    .getOrCreate()

    sppMLTrainingDao:SppMLTrainingDao = SppMLTrainingDao.SppMLTrainingDao(spark)
    trainingStartDate = datetime.strftime(datetime.strptime(pScoreStartDate, '%Y-%m-%d') - timedelta(days=trainingDataDays), '%Y-%m-%d')
    trainingDataCtx = {
        'trainingStartDate': trainingStartDate
        , 'trainingEndDate': pScoreEndDate
        , 'exchange': exchange
        , 'index': index
        , 'exchangeCode': exchangeCode
    }

    trainingData:dict = sppMLTrainingDao.loadTrainingData(trainingDataCtx)
    transformer:SppTrainingDataTransformer = SppTrainingDataTransformer.SppTrainingDataTransformer(trainingData, trainingDataCtx)
    trainingData = transformer.transform()

    pScoreStartDate = datetime.strptime(pScoreStartDate, '%Y-%m-%d')
    pScoreEndDate = datetime.strptime(pScoreEndDate, '%Y-%m-%d')

    modelCache = {}
    for d in rrule(DAILY, dtstart=pScoreStartDate, until=pScoreEndDate):
        pScoreDate = datetime.strftime(d, '%Y-%m-%d')
        trainingEndDate = pScoreDate
        trainingStartDate = datetime.strftime(datetime.strptime(trainingEndDate, '%Y-%m-%d') - timedelta(days=trainingDataDays), '%Y-%m-%d')
        trainingDataForTraining = extractTrainingData(trainingData, trainingStartDate, trainingEndDate)
        ctx = {'exchange': exchange
            , 'pScoreDate': pScoreDate
            , 'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'index': index
            , 'exchangeCode': exchangeCode
            , 'forecastDays': forecastDays
            , 'forecastor': forecastor
            , 'trainingDataForTraining': trainingDataForTraining
            , 'cacheAndRetrainModel': True
            , 'modelCache':modelCache}
        runSpp(ctx, sppMLTrainingDao, spark)
        modelCache = ctx['modelCache']

if __name__ == '__main__':
    main(sys.argv[1:])