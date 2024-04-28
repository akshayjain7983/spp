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
from .util import util

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
    forecastDays = int(args[2])
    exchange = args[3]
    index = args[4]
    segment = args[5]
    dataHistoryMonths = args[6]
    forecastor = args[7]
    exchangeCode = args[8] if len(args) >= 9 else None

    trainingDataDays = 31*int(dataHistoryMonths)

    sppMLTrainingDao:SppMLDao = SppMLDao.SppMLDao(config)
    holidays = sppMLTrainingDao.loadHolidays({
                                                'exchange': exchange
                                                , 'segment': segment
                                                , 'currentDate': pScoreEndDate+timedelta(days=365)
                                                , 'days': trainingDataDays*2
                                            })

    trainingStartDate = util.previous_business_date(pScoreStartDate, trainingDataDays, holidays)
    trainingDataCtx = {'trainingStartDate': trainingStartDate
        , 'trainingEndDate': pScoreEndDate
        , 'exchange': exchange
        , 'index': index
        , 'exchangeCode': exchangeCode
        , 'segment': segment
    }

    trainingData:dict = sppMLTrainingDao.loadTrainingData(trainingDataCtx)
    transformer:SppTrainingDataTransformer = SppTrainingDataTransformer.SppTrainingDataTransformer(trainingData, trainingDataCtx)
    trainingData = transformer.transform()

    modelCache = {}
    for d in rrule(DAILY, dtstart=pScoreStartDate, until=pScoreEndDate):
        pScoreDate = d.date()
        if(util.is_holiday(pScoreDate, holidays)):
            continue
        trainingEndDate = pScoreDate
        trainingStartDate = util.previous_business_date(trainingEndDate, trainingDataDays, holidays)
        trainingDataForTraining = extractTrainingData(trainingData, trainingStartDate, trainingEndDate)
        ctx = {'exchange': exchange
            , 'pScoreDate': pScoreDate
            , 'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'index': index
            , 'segment': segment
            , 'holidays': holidays
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