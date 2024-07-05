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

    return childProcess

def waitForChildProcesses(childProcesses:[]):
    returnCodes = []
    returnCodes.clear()
    while(len(returnCodes) < len(childProcesses)):
        time.sleep(1)
        for cp in childProcesses:
            rc = cp.poll()
            if(rc != None):
                returnCodes.append((rc, cp))

    for rc in returnCodes:
        if(rc[0] != 0):
            raise ChildProcessError('Child process exited with error for '+rc[1].args[-1]+' : '+str(rc[0]))

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
    forecastIndex = bool(args[8])
    exchangeCodes = [ec for ec in args[9].split(',')] if len(args) >= 10 else None

    sppMLTrainingDao: SppMLDao = SppMLDao.SppMLDao(config)

    if(exchangeCodes == None):
        exchangeCodes = sppMLTrainingDao.loadSecurityExchangeCodes({
                                                                    'exchange':exchange
                                                                    , 'exchangeCodes': None
                                                                    })
        exchangeCodes = exchangeCodes.loc[0:]['exchange_code'].values

    trainingDataDays = 31*int(dataHistoryMonths)

    holidays = sppMLTrainingDao.loadHolidays({
                                                'exchange': exchange
                                                , 'segment': segment
                                                , 'currentDate': pScoreEndDate+timedelta(days=365)
                                                , 'days': trainingDataDays*2
                                            })

    pScoreBusinessDates = util.business_date_range(pScoreStartDate, pScoreEndDate, holidays)
    childProcesses = []

    for pScoreDate in pScoreBusinessDates:

        print("Starting for date: "+datetime.strftime(pScoreDate, '%Y-%m-%d'))

        max_processes = int(os.cpu_count() * 0.5) + 2 if (pScoreDate > pScoreBusinessDates[0]) else 1 #less parallel for first date since many models are trained during first time tuning
        process_count = 0
        childProcesses.clear()

        if(forecastIndex):
            # first submit for index
            ctxIndex = {'trainerInvocationMode': 'index'
                , 'exchange': exchange
                , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                , 'subjectCode': index
                , 'segment': segment
                , 'forecastDays': str(forecastDays)
                , 'forecastor': forecastor
                , 'dataHistoryMonths': dataHistoryMonths}

            childProcessIdx = triggerSubProcessForForecasting(ctxIndex)
            childProcesses.append(childProcessIdx)
            process_count += 1

            print("Triggered for subject: " + index + ' using ctx:'+str(ctxIndex))

            if (process_count == max_processes):
                waitForChildProcesses(childProcesses)
                process_count = 0
                childProcesses.clear()

        # next do for securities
        for ec in exchangeCodes:
            ctxSecurity = {'trainerInvocationMode': 'security'
                , 'exchange': exchange
                , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                , 'subjectCode': ec
                , 'segment': segment
                , 'forecastDays': str(forecastDays)
                , 'forecastor': forecastor
                , 'dataHistoryMonths': dataHistoryMonths}

            childProcessSec = triggerSubProcessForForecasting(ctxSecurity)
            childProcesses.append(childProcessSec)
            process_count += 1

            print("Triggered for subject: " + ec + ' using ctx:' + str(ctxSecurity))

            if(process_count == max_processes):
                waitForChildProcesses(childProcesses)
                process_count = 0
                childProcesses.clear()

        # finish off this date in case any left to finish before moving to next date
        if(len(childProcesses)>0):
            waitForChildProcesses(childProcesses)

        print("PScore for date: " + datetime.strftime(pScoreDate, '%Y-%m-%d'))
        #finally trigger pScore calc for date
        sppMLTrainingDao.saveForecastPScore({'exchange': exchange
                , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                , 'index': index
                , 'segment': segment
                , 'forecastPeriod': str(forecastDays)+'d'
                , 'forecastor': forecastor}
            , exchangeCodes)

        print("Finished for date: " + datetime.strftime(pScoreDate, '%Y-%m-%d'))

if __name__ == '__main__':
    main(sys.argv[1:])