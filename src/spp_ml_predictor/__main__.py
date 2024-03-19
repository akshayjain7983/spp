import pandas as pd
from .dao import SppMLTrainingDao
from .dao import SppMLDao
from .dao import QueryFiles
from .trainer import SppTrainer
from .trainer import SppTrainingDataTransformer
import sys
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
from .config.ConfigReader import readConfig

def runSpp(ctx:dict, sppMLTrainingDao: SppMLTrainingDao):

    sppTrainer:SppTrainer = SppTrainer.SppTrainer(sppMLTrainingDao, ctx)
    sppTrainer.train()


def extractTrainingData(trainingData:dict, trainingStartDate, trainingEndDate):

    indexLevelsPdf:pd.DataFrame = trainingData['indexLevelsPdf']
    indexLevelsPdfForTraining = indexLevelsPdf[(indexLevelsPdf.date >= trainingStartDate) & (indexLevelsPdf.date <= trainingEndDate)]
    securityPricesPdf:pd.DataFrame = trainingData['securityPricesPdf']
    securityPricesPdfForTraining = securityPricesPdf[(securityPricesPdf.date >= trainingStartDate) & (securityPricesPdf.date <= trainingEndDate)]

    trainingDataCopy = trainingData.copy()
    trainingDataCopy['indexLevelsPdf'] = indexLevelsPdfForTraining
    trainingDataCopy['securityPricesPdf'] = securityPricesPdfForTraining
    return trainingDataCopy

def main(args):
    config = readConfig(configFilename = 'spp_ml_predictor/config.ini')
    QueryFiles.load()

    pScoreStartDate = datetime.strptime(args[0], '%Y-%m-%d').date()
    pScoreEndDate = datetime.strptime(args[1], '%Y-%m-%d').date()
    forecastDays = [int(e) for e in args[2].split(',')]
    forecastDays.append(0)
    forecastDays.sort()
    exchange = args[3]
    index = args[4]
    dataHistoryMonths = args[5]
    forecastor = args[6]
    exchangeCode = args[7] if len(args) >= 8 else None

    trainingDataDays = 31*int(dataHistoryMonths)

    sppMLTrainingDao:SppMLDao = SppMLDao.SppMLDao(config)
    trainingStartDate = pScoreStartDate - timedelta(days=trainingDataDays)
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

    modelCache = {}
    for d in rrule(DAILY, dtstart=pScoreStartDate, until=pScoreEndDate):
        pScoreDate = d.date()
        trainingEndDate = pScoreDate
        trainingStartDate = trainingEndDate - timedelta(days=trainingDataDays)
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
        runSpp(ctx, sppMLTrainingDao)
        modelCache = ctx['modelCache']

if __name__ == '__main__':
    main(sys.argv[1:])