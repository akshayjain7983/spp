from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime, timedelta
import time

from dateutil.rrule import rrule, DAILY

import pyspark.sql as ps
import pyspark.sql.functions as psf

from ..dao import SppMLTrainingDao
from ..trainer.SppIndexForecastTask import SppIndexForecastTask
from ..trainer.SppSecurityForecastTask import SppSecurityForecastTask
from ..util.SppUtil import fillGapsOrdered
from pyspark.sql.types import DoubleType


class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def __submitForSppSecurityForecastTask__(self
                                             , forecastIndexReturns:ps.DataFrame
                                             , securityPricesPdf:ps.DataFrame
                                             , interestRatesPdf:ps.DataFrame
                                             , inflationRatesPdf:ps.DataFrame) -> ps.DataFrame:


        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        pScoreDate = datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d')
        securityPricesPdfLocal = securityPricesPdf[(securityPricesPdf.date >= startDate) & (securityPricesPdf.date <= pScoreDate)]
        spark: ps.SparkSession = self.ctx['spark']
        extendedData = [d for d in rrule(DAILY, dtstart=startDate, until=pScoreDate)]
        securityPricesPdfLocal = fillGapsOrdered(securityPricesPdfLocal, spark, 'date', extendedData)
        securityPricesPdfLocal = securityPricesPdfLocal.dropna().sort('date')
        
        xtraDataPdf: ps.DataFrame = interestRatesPdf.select('date', 'rate').withColumnRenamed('rate', 'repo')
        xtraDataPdf = xtraDataPdf.join(inflationRatesPdf.select('date', 'rate').withColumnRenamed('rate', 'inflation'), 'date')
        xtraDataPdf = xtraDataPdf.join(securityPricesPdfLocal.select('date', 'candlestickMovementReal'), 'date', 'leftouter')

        sppTrainingTask = SppSecurityForecastTask(forecastIndexReturns, securityPricesPdfLocal, self.ctx, xtraDataPdf)
        forecast = sppTrainingTask.forecast()
        # self.sppMLTrainingDao.saveForecastPScore(forecast, self.ctx)
        return forecast;

    def __submitForSppIndexForecastTask__(self
                                          , indexLevelsPdf:ps.DataFrame
                                          , interestRatesPdf:ps.DataFrame
                                          , inflationRatesPdf:ps.DataFrame) -> ps.DataFrame:
        
        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        pScoreDate = datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d')
        indexLevelsPdfLocal = indexLevelsPdf[(indexLevelsPdf.date >= startDate) & (indexLevelsPdf.date <= pScoreDate)]
        spark: ps.SparkSession = self.ctx['spark']
        extendedData = [d for d in rrule(DAILY, dtstart=startDate, until=pScoreDate)]
        indexLevelsPdfLocal = fillGapsOrdered(indexLevelsPdfLocal, spark, 'date', extendedData)
        indexLevelsPdfLocal = indexLevelsPdfLocal.dropna().sort('date')
        
        xtraDataPdf: ps.DataFrame = interestRatesPdf.select(psf.col('date'), psf.col('rate').name('repo').cast(DoubleType()))
        xtraDataPdf = xtraDataPdf.join(inflationRatesPdf.select(psf.col('date'), psf.col('rate').name('inflation').cast(DoubleType())), 'date')
        xtraDataPdf = xtraDataPdf.join(indexLevelsPdfLocal.select('date', 'candlestickMovementReal'), 'date', 'leftouter')
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

        multithread:bool = exchangeCodePdf['exchangeCode'].size > 1

        if(multithread):
            futures = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                for ec in exchangeCodePdf['exchangeCode']:
                    future = executor.submit(self.__submitForSppSecurityForecastTask__, forecastIndexReturns
                                             , securityPricesPdf[securityPricesPdf['exchangeCode'] == ec]
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
        print("SppTrainer - "+self.ctx['pScoreDate']+" - Time taken:" + str(endT - startT) + " secs")

    def setupInterestRates(self, interestRatesPdf:ps.DataFrame) -> ps.DataFrame:

        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        forecastDate = endDate + timedelta(days=self.ctx['forecastDays'][-1])

        # firt get interest rates for training period
        interestRatesPdf = interestRatesPdf[(interestRatesPdf.date >= startDate) & (interestRatesPdf.date <= endDate)]
        interestRatesPdf = interestRatesPdf.sort('date')

        # assume interest rates remains constant for forecast period
        spark: ps.SparkSession = self.ctx['spark']
        extendedData = [d for d in rrule(DAILY, dtstart=startDate, until=forecastDate)]
        interestRatesPdf = fillGapsOrdered(interestRatesPdf, spark, 'date', extendedData)
        
        # finally make rates as fractions
        interestRatesPdf = interestRatesPdf.withColumn('rate', interestRatesPdf['rate']/100)

        return interestRatesPdf

    def setupInflationRates(self, inflationRatesPdf:ps.DataFrame) -> ps.DataFrame:

        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        forecastDate = endDate + timedelta(days=self.ctx['forecastDays'][-1])

        # firt get inflation rates for training period
        inflationRatesPdf = inflationRatesPdf[(inflationRatesPdf.date >= startDate) & (inflationRatesPdf.date <= endDate)]
        inflationRatesPdf = inflationRatesPdf.sort('date')

        # assume inflation rates remains constant for forecast period
        spark: ps.SparkSession = self.ctx['spark']
        extendedData = [d for d in rrule(DAILY, dtstart=startDate, until=forecastDate)]
        inflationRatesPdf = fillGapsOrdered(inflationRatesPdf, spark, 'date', extendedData)
        
        # finally make rates as fractions
        inflationRatesPdf = inflationRatesPdf.withColumn('rate', inflationRatesPdf['rate']/100)

        return inflationRatesPdf
    