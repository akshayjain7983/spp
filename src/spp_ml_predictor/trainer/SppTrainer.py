from ..dao import SppMLDao
import pandas as pd
from ..trainer.SppSecurityForecastTask import SppSecurityForecastTask
from ..trainer.SppIndexForecastTask import SppIndexForecastTask
from concurrent.futures import *
from datetime import datetime, timedelta
import time
from ..util import util
import os

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def __submitForSppSecurityForecastTask__(self
                                             , forecastIndexReturns:pd.DataFrame
                                             , securityPricesPdf:pd.DataFrame
                                             , interestRatesPdf:pd.DataFrame
                                             , inflationRatesPdf:pd.DataFrame) -> pd.DataFrame:

        securityPricesPdfLocal = securityPricesPdf.copy()
        securityPricesPdfLocal.sort_index(inplace=True)
        trainingDataForTraining:dict = self.ctx['trainingDataForTraining']
        indexLevelsPdf = trainingDataForTraining['indexLevelsPdf']
        securityPricesPdfLocal = securityPricesPdfLocal.reindex(indexLevelsPdf.index, method='ffill')

        xtraDataPdf:pd.DataFrame = interestRatesPdf.drop(['institution', 'rate_type'], axis=1)
        xtraDataPdf.rename(columns={"rate": "repo"}, inplace=True)
        xtraDataPdf['inflation'] = inflationRatesPdf['rate']
        xtraDataPdf['candlestickMovement'] = securityPricesPdfLocal['candlestickMovement']

        sppTrainingTask = SppSecurityForecastTask(forecastIndexReturns, securityPricesPdfLocal, self.ctx, xtraDataPdf)
        forecast = sppTrainingTask.forecast()
        # self.sppMLTrainingDao.saveForecastPScore(forecast, self.ctx)
        return forecast;

    def __submitForSppIndexForecastTask__(self
                                          , indexLevelsPdf:pd.DataFrame
                                          , interestRatesPdf:pd.DataFrame
                                          , inflationRatesPdf:pd.DataFrame) -> pd.DataFrame:

        indexLevelsPdfLocal = indexLevelsPdf.copy()
        indexLevelsPdfLocal.sort_index(inplace=True)

        xtraDataPdf = interestRatesPdf.drop(['institution', 'rate_type'], axis=1)
        xtraDataPdf.rename(columns={"rate": "repo"}, inplace=True)
        xtraDataPdf['inflation'] = inflationRatesPdf['rate']
        xtraDataPdf['candlestickMovement'] = indexLevelsPdfLocal['candlestickMovement']

        sppTrainingTask = SppIndexForecastTask(indexLevelsPdfLocal, self.ctx, xtraDataPdf)
        forecast = sppTrainingTask.forecast()
        return forecast;


    def train(self):

        startT = time.time()

        trainingDataForTraining:dict = self.ctx['trainingDataForTraining']
        indexLevelsPdf = trainingDataForTraining['indexLevelsPdf']
        securityPricesPdf = trainingDataForTraining['securityPricesPdf']
        interestRatesPdf = trainingDataForTraining['interestRatesPdf']
        inflationRatesPdf = trainingDataForTraining['inflationRatesPdf']
        exchangeCodePdf = trainingDataForTraining['exchangeCodePdf']

        interestRatesPdf = self.setupInterestRates(interestRatesPdf)
        inflationRatesPdf = self.setupInflationRates(inflationRatesPdf)
        forecastIndexReturns = self.__submitForSppIndexForecastTask__(indexLevelsPdf, interestRatesPdf, inflationRatesPdf)

        minSecurityPricesData = int(indexLevelsPdf.shape[0] * 0.99)
        securityPricesPdf =securityPricesPdf.groupby('exchange_code').filter(lambda x: len(x) > minSecurityPricesData)
        
        multithread:bool = exchangeCodePdf['exchange_code'].size > 1

        if(multithread):
            futures = []
            max_workers = int(os.cpu_count()*0.5)+1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for ec in exchangeCodePdf['exchange_code']:
                    securityPricesPdfForEc = securityPricesPdf[securityPricesPdf['exchange_code'] == ec]
                    if(securityPricesPdfForEc.shape[0] > 0):
                        future = executor.submit(self.__submitForSppSecurityForecastTask__, forecastIndexReturns
                                                , securityPricesPdfForEc
                                                , interestRatesPdf, inflationRatesPdf)
                        futures.append(future)

            for f in futures:
                try:
                    print(f.result())
                except Exception:
                    print(f.exception())

        else:
            forecast = self.__submitForSppSecurityForecastTask__(forecastIndexReturns, securityPricesPdf, interestRatesPdf, inflationRatesPdf)
            print(forecast)

        endT = time.time()
        print("SppTrainer - "+datetime.strftime(self.ctx['pScoreDate'], '%Y-%m-%d`1')+" - Time taken:" + str(endT - startT) + " secs")

    def setupInterestRates(self, interestRatesPdf:pd.DataFrame) -> pd.DataFrame:

        startDate = self.ctx['trainingStartDate']
        endDate = self.ctx['trainingEndDate']
        forecastDate = util.next_business_date(endDate, self.ctx['forecastDays'], self.ctx['holidays'])

        # firt get interest rates for training period
        interestRatesReindexPdf = pd.date_range(start=startDate, end=endDate)
        interestRatesPdf = interestRatesPdf[interestRatesPdf.index.isin(interestRatesReindexPdf)]
        interestRatesPdf = interestRatesPdf.reindex(interestRatesReindexPdf, method='ffill')
        interestRatesPdf.sort_index(inplace=True)

        # assume interest rates remains constant for forecast period
        interestRatesReindexPdf = pd.date_range(start=startDate, end=forecastDate)
        interestRatesPdf = interestRatesPdf.reindex(interestRatesReindexPdf, method='ffill')
        interestRatesPdf.sort_index(inplace=True)

        # finally make rates as fractions
        interestRatesPdf['rateFraction'] = interestRatesPdf['rate'] / 100
        interestRatesPdf.drop(['rate'], axis=1, inplace=True)
        interestRatesPdf.rename(columns={"rateFraction": "rate"}, inplace=True)

        return interestRatesPdf

    def setupInflationRates(self, inflationRatesPdf:pd.DataFrame) -> pd.DataFrame:

        startDate = self.ctx['trainingStartDate']
        endDate = self.ctx['trainingEndDate']
        forecastDate = util.next_business_date(endDate, self.ctx['forecastDays'], self.ctx['holidays'])

        # firt get inflation rates for training period
        inflationRatesReindexPdf = pd.date_range(start=startDate, end=endDate)
        inflationRatesPdf = inflationRatesPdf[inflationRatesPdf.index.isin(inflationRatesReindexPdf)]
        inflationRatesPdf = inflationRatesPdf.reindex(inflationRatesReindexPdf, method='bfill')
        inflationRatesPdf.sort_index(inplace=True)

        # assume inflation rates remains constant for forecast period
        inflationRatesReindexPdf = pd.date_range(start=startDate, end=forecastDate)
        inflationRatesPdf = inflationRatesPdf.reindex(inflationRatesReindexPdf, method='ffill')
        inflationRatesPdf.sort_index(inplace=True)

        #finally make rates as fractions
        inflationRatesPdf['rateFraction'] = inflationRatesPdf['rate']/100
        inflationRatesPdf.drop(['rate'], axis=1, inplace=True)
        inflationRatesPdf.rename(columns={"rateFraction": "rate"}, inplace=True)

        return inflationRatesPdf