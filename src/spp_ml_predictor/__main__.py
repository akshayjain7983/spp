import os
import subprocess
import sys
import time
import logging

from .dao import SppMLDao
from .dao import QueryFiles
from datetime import datetime, timedelta
from .config.ConfigReader import readConfig
from .util import util
from .trainer import trainer
import os

logger = logging.getLogger('spp')

def triggerSubProcessForForecasting(ctx:dict):
    pypath = sys.executable
    env = os.environ.copy()
    useGpu = ctx['useGpu']
    if(not useGpu):
        env['CUDA_VISIBLE_DEVICES'] = '-1'

    childProcess = subprocess.Popen([pypath
                                        , '-m'
                                        , 'spp_ml_predictor.trainer'
                                        , ctx['config']
                                        , ctx['trainerInvocationMode']
                                        , ctx['pScoreDate']
                                        , ctx['forecastDays']
                                        , ctx['exchange']
                                        , ctx['dataHistoryMonths']
                                        , ctx['forecastor']
                                        , ctx['segment']
                                        , ctx['subjectCode']
                                     ], env=env)

    return childProcess

def triggerCurrentProcessForForecasting(ctx:dict):
    trainer.main([
        ctx['config']
        , ctx['trainerInvocationMode']
        , ctx['pScoreDate']
        , ctx['forecastDays']
        , ctx['exchange']
        , ctx['dataHistoryMonths']
        , ctx['forecastor']
        , ctx['segment']
        , ctx['subjectCode']
        ])


def waitForChildProcesses(childProcesses:[]):
    returnCodes = []
    failedSecurityProcesses = {}
    returnCodes.clear()
    while(len(returnCodes) < len(childProcesses)):
        time.sleep(1)
        for cp in childProcesses:
            rc = cp.poll()
            if(rc != None):
                returnCodes.append((rc, cp))

    for rc in returnCodes:
        if(rc[0] != 0):
            failedSecurityProcesses[rc[1].args[-1]] = rc[0]
            logger.error('Child process exited with error for '+rc[1].args[-1]+' : '+str(rc[0]))

    return failedSecurityProcesses

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
    multiprocessing = True #len(exchangeCodes) > 1
    maxProcessesConfig = int(config['ml-params.max-process'])
    config = 'None' if(multiprocessing) else config
    failedSecurityProcesses = {}

    for pScoreDate in pScoreBusinessDates:

        logger.info("Starting for date: "+datetime.strftime(pScoreDate, '%Y-%m-%d'))

        max_processes = maxProcessesConfig if (pScoreDate > pScoreBusinessDates[0]) else 1 #less parallel for first date since many models are trained during first time tuning
        process_count = 0
        childProcesses.clear()

        if(forecastIndex):
            # first submit for index
            ctxIndex = {'config': config
                , 'trainerInvocationMode': 'index'
                , 'exchange': exchange
                , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                , 'subjectCode': index
                , 'segment': segment
                , 'forecastDays': str(forecastDays)
                , 'forecastor': forecastor
                , 'dataHistoryMonths': dataHistoryMonths
                , 'useGpu': max_processes == 1}

            if(multiprocessing):
                childProcessIdx = triggerSubProcessForForecasting(ctxIndex)
                childProcesses.append(childProcessIdx)
                process_count += 1
                logger.info("Triggered for subject: " + index + ' using ctx:' + str(ctxIndex))

                if (process_count == max_processes):
                    failedSecurityProcessesT1 = waitForChildProcesses(childProcesses)
                    if(len(failedSecurityProcessesT1)>0):
                        raise ChildProcessError('Failed to predict for index. Cannot process further')
                    process_count = 0
                    childProcesses.clear()

            else:
                triggerCurrentProcessForForecasting(ctxIndex)

        # next do for securities
        for ec in exchangeCodes:

            if(ec in failedSecurityProcesses):
                continue #do not execute for any failed security for any past date

            ctxSecurity = {'config': config
                , 'trainerInvocationMode': 'security'
                , 'exchange': exchange
                , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                , 'subjectCode': ec
                , 'segment': segment
                , 'forecastDays': str(forecastDays)
                , 'forecastor': forecastor
                , 'dataHistoryMonths': dataHistoryMonths
                , 'useGpu': max_processes == 1}

            if(multiprocessing):
                childProcessSec = triggerSubProcessForForecasting(ctxSecurity)
                childProcesses.append(childProcessSec)
                process_count += 1

                logger.info("Triggered for subject: " + ec + ' using ctx:' + str(ctxSecurity))

                if(process_count == max_processes):
                    failedSecurityProcessesT1 = waitForChildProcesses(childProcesses)
                    if (len(failedSecurityProcessesT1) > 0):
                        failedSecurityProcesses.update(failedSecurityProcessesT1)
                    process_count = 0
                    childProcesses.clear()
            else:
                triggerCurrentProcessForForecasting(ctxSecurity)

        # finish off this date in case any left to finish before moving to next date
        if(multiprocessing and len(childProcesses)>0):
            failedSecurityProcessesT1 = waitForChildProcesses(childProcesses)
            if (len(failedSecurityProcessesT1) > 0):
                failedSecurityProcesses.update(failedSecurityProcessesT1)

        logger.info("PScore for date: " + datetime.strftime(pScoreDate, '%Y-%m-%d'))
        #finally trigger pScore calc for date
        sppMLTrainingDao.saveForecastPScore({'exchange': exchange
                , 'pScoreDate': datetime.strftime(pScoreDate, '%Y-%m-%d')
                , 'index': index
                , 'segment': segment
                , 'forecastPeriod': str(forecastDays)+'d'
                , 'forecastor': forecastor}
            , exchangeCodes)

        logger.info("Finished for date: " + datetime.strftime(pScoreDate, '%Y-%m-%d'))

if __name__ == '__main__':
    main(sys.argv[1:])