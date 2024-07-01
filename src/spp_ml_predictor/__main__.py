import os
import subprocess
import sys
import pathlib
import time

import pandas as pd
from .dao import SppMLDao
from .dao import QueryFiles
from .trainer import SppTrainer
from .trainer import SppTrainingDataTransformer
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
from .config.ConfigReader import readConfig
from .util import util
from concurrent.futures import ThreadPoolExecutor

def triggerSubProcessForForecasting(ctx:dict):
    pypath = sys.executable
    childProcess = subprocess.Popen([pypath
                                        , '-m'
                                        , 'spp_ml_predictor.trainer'
                                        , ctx['trainerInvocationMode']
                                        , ctx['pScoreDate']
                                        , ctx['forecastDays']
                                        , ctx['exchange']
                                        , ctx['dataHistoryMonths']
                                        , ctx['forecastor']
                                        , ctx['segment']
                                        , ctx['subjectCode']
                                     ])

    while(childProcess.poll() == None):
        time.sleep(1)

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
    exchangeCodes = [ec for ec in args[8].split(',')] if len(args) >= 9 else None

    trainingDataDays = 31*int(dataHistoryMonths)

    sppMLTrainingDao:SppMLDao = SppMLDao.SppMLDao(config)
    holidays = sppMLTrainingDao.loadHolidays({
                                                'exchange': exchange
                                                , 'segment': segment
                                                , 'currentDate': pScoreEndDate+timedelta(days=365)
                                                , 'days': trainingDataDays*2
                                            })

    # first do index returns for whole date range
    for d in rrule(DAILY, dtstart=pScoreStartDate, until=pScoreEndDate):
        pScoreDate = d.date()
        if(util.is_holiday(pScoreDate, holidays)):
            continue

        ctx = {'trainerInvocationMode': 'index'
            , 'exchange': exchange
            , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
            , 'subjectCode': index
            , 'segment': segment
            , 'forecastDays': str(forecastDays)
            , 'forecastor': forecastor
            , 'dataHistoryMonths': dataHistoryMonths}

        triggerSubProcessForForecasting(ctx)

    # next do security returns for whole date range
    for d in rrule(DAILY, dtstart=pScoreStartDate, until=pScoreEndDate):
        pScoreDate = d.date()
        if(util.is_holiday(pScoreDate, holidays)):
            continue

        multi = len(exchangeCodes) > 1

        if(multi):
            max_workers = int(os.cpu_count() * 0.5) + 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for ec in exchangeCodes:
                    ctx = {'trainerInvocationMode': 'security'
                        , 'exchange': exchange
                        , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                        , 'subjectCode': ec
                        , 'segment': segment
                        , 'forecastDays': str(forecastDays)
                        , 'forecastor': forecastor
                        , 'dataHistoryMonths': dataHistoryMonths}

                    executor.submit(triggerSubProcessForForecasting, ctx)
        else:
            triggerSubProcessForForecasting(ctx)

if __name__ == '__main__':
    main(sys.argv[1:])