import pandas as pd
from ..dao import SppMLTrainingDao
from ..dao.SppMLDao import SppMLDao
from ..dao import QueryFiles
from ..trainer.SppTrainer import SppTrainer
import sys
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
from ..config.ConfigReader import readConfig
from ..util import util


def main(args):
    config = readConfig(configFilename = 'spp_ml_predictor/config.ini')
    QueryFiles.load()

    trainerInvocationMode = args[0]
    pScoreDate = datetime.strptime(args[1], '%Y-%m-%d').date()
    forecastDays = int(args[2])
    exchange = args[3]
    dataHistoryMonths = args[4]
    forecastor = args[5]
    segment = args[6]
    index = args[7] if trainerInvocationMode == 'index' else None
    exchangeCode = args[7] if trainerInvocationMode == 'security' else None

    if(not (trainerInvocationMode == 'security' or trainerInvocationMode == 'index')):
        raise KeyError('trainerInvocationMode can be one of "security" or "index"')

    trainingDataDays = 31 * int(dataHistoryMonths)
    sppMLTrainingDao: SppMLDao = SppMLDao(config)
    holidays = sppMLTrainingDao.loadHolidays({
        'exchange': exchange
        , 'segment': segment
        , 'currentDate': pScoreDate + timedelta(days=365)
        , 'days': trainingDataDays * 2
    })

    if(util.is_holiday(pScoreDate, holidays)):
        raise KeyError('pScoreDate cannot be a holiday')

    trainingStartDate = util.previous_business_date(pScoreDate, trainingDataDays, holidays)
    trainingEndDate = pScoreDate

    if(trainerInvocationMode == 'index'):
        trainingDataCtx = {'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'exchange': exchange
            , 'index': index
            , 'trainerInvocationMode': trainerInvocationMode
        }
        trainingData: dict = sppMLTrainingDao.loadTrainingDataForIndex(trainingDataCtx)
        trainingData = util.transformForIndex(trainingData)
        ctx = {'exchange': exchange
            , 'pScoreDate': pScoreDate
            , 'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'index': index
            , 'holidays': holidays
            , 'forecastDays': forecastDays
            , 'forecastor': forecastor
            , 'trainingDataForTraining': trainingData
            , 'config': config}

        sppTrainer: SppTrainer = SppTrainer(sppMLTrainingDao, ctx)
        sppTrainer.forecastForIndex()

    elif(trainerInvocationMode == 'security'):
        trainingDataCtx = {'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'exchange': exchange
            , 'exchangeCodes': [exchangeCode]
            , 'trainerInvocationMode': trainerInvocationMode
            }
        trainingData: dict = sppMLTrainingDao.loadTrainingDataForSecurity(trainingDataCtx)
        trainingData = util.transformForSecurity(trainingData)
        ctx = {'exchange': exchange
            , 'pScoreDate': pScoreDate
            , 'trainingStartDate': trainingStartDate
            , 'trainingEndDate': trainingEndDate
            , 'exchangeCode': exchangeCode
            , 'holidays': holidays
            , 'forecastDays': forecastDays
            , 'forecastor': forecastor
            , 'trainingDataForTraining': trainingData
            , 'config': config}

        sppTrainer: SppTrainer = SppTrainer(sppMLTrainingDao, ctx)
        sppTrainer.forecastForSecurity()


if __name__ == '__main__':
    main(sys.argv[1:])