from ..dao import SppMLTrainingDao
import pandas as pd
from ..trainer.SppSecurityForecastTask import SppSecurityForecastTask
from ..trainer.SppIndexForecastTask import SppIndexForecastTask
from concurrent.futures import *
from datetime import datetime, timedelta
import time

class SppTrainer:
    def __init__(self, sppMLTrainingDao:SppMLTrainingDao, ctx:dict):
        self.sppMLTrainingDao = sppMLTrainingDao
        self.ctx = ctx

    def __submitForSppSecurityForecastTask__(self
                                             , forecastIndexReturns:pd.DataFrame
                                             , securityReturnsPdf:pd.DataFrame
                                             , interestRatesPdf:pd.DataFrame
                                             , inflationRatesPdf:pd.DataFrame) -> pd.DataFrame:

        securityReturnsPdfLocal = securityReturnsPdf.copy()
        securityReturnsPdfLocal['datetime'] = pd.to_datetime(securityReturnsPdfLocal['date'])
        securityReturnsPdfLocal.set_index("datetime", inplace=True, drop=True)
        securityReturnsPdfLocal.sort_index(inplace=True)
        securityReturnsReindexPdf = pd.date_range(start=datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d'),
                                                  end=datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d'),
                                                  inclusive="both")
        securityReturnsPdfLocal = securityReturnsPdfLocal.reindex(securityReturnsReindexPdf, method='ffill')
        securityReturnsPdfLocal.dropna(inplace=True) #sometimes ffill with reindex may result in nan at begining so drop those
        securityReturns90DSeries = [d.get('90D').get('return') for d in securityReturnsPdfLocal["returns"]]
        securityReturnsPdfLocal.drop("returns", axis=1, inplace=True)
        securityReturnsPdfLocal["securityReturns90D"] = securityReturns90DSeries

        xtraDataPdf = interestRatesPdf.drop(['_id', 'date', 'institution', 'rateType'], axis=1)
        xtraDataPdf.rename(columns={"rate": "repo"}, inplace=True)
        xtraDataPdf['inflation'] = inflationRatesPdf['rate']

        sppTrainingTask = SppSecurityForecastTask(forecastIndexReturns, securityReturnsPdfLocal, self.ctx, xtraDataPdf)
        forecast = sppTrainingTask.forecast()
        self.sppMLTrainingDao.saveForecastPScore(forecast)
        return forecast;

    def __submitForSppIndexForecastTask__(self
                                          , indexReturnsPdf:pd.DataFrame
                                          , interestRatesPdf:pd.DataFrame
                                          , inflationRatesPdf:pd.DataFrame) -> pd.DataFrame:

        indexReturnsPdfLocal = indexReturnsPdf.copy()
        indexReturnsPdfLocal['datetime'] = pd.to_datetime(indexReturnsPdfLocal['date'])
        indexReturnsPdfLocal.set_index("datetime", inplace=True, drop=True)
        indexReturnsPdfLocal.sort_index(inplace=True)
        indexReturnsReindexPdf = pd.date_range(start=datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d'),
                                               end=datetime.strptime(self.ctx['pScoreDate'], '%Y-%m-%d'),
                                               inclusive="both")
        indexReturnsPdfLocal = indexReturnsPdfLocal.reindex(indexReturnsReindexPdf, method='ffill')
        indexReturnsPdfLocal.dropna(inplace=True) #sometimes ffill with reindex may result in nan at begining so drop those
        indexReturns90DSeries = [d.get('90D').get('return') for d in indexReturnsPdfLocal["returns"]]
        indexReturnsPdfLocal.drop("returns", axis=1, inplace=True)
        indexReturnsPdfLocal["indexReturns90D"] = indexReturns90DSeries

        xtraDataPdf = interestRatesPdf.drop(['_id', 'date', 'institution', 'rateType'], axis=1)
        xtraDataPdf.rename(columns={"rate": "repo"}, inplace=True)
        xtraDataPdf['inflation'] = inflationRatesPdf['rate']


        sppTrainingTask = SppIndexForecastTask(indexReturnsPdfLocal, self.ctx, xtraDataPdf)
        forecast = sppTrainingTask.forecast()
        return forecast;

    def train(self):

        startT = time.time()
        indexReturnsPdf:pd.DataFrame = self.sppMLTrainingDao.loadIndexReturns(self.ctx)
        interestRatesPdf:pd.DataFrame = self.sppMLTrainingDao.loadInterestRates("Reserve Bank of India", "repo")
        interestRatesPdf = self.setupInterestRates(interestRatesPdf)
        inflationRatesPdf: pd.DataFrame = self.sppMLTrainingDao.loadInflationRates("Reserve Bank of India", "CPI - YoY - General")
        inflationRatesPdf = self.setupInflationRates(inflationRatesPdf)
        forecastIndexReturns = self.__submitForSppIndexForecastTask__(indexReturnsPdf, interestRatesPdf, inflationRatesPdf)

        exchangeCodeDf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityExchangeCodes(self.ctx)
        exchangeCodeDf = exchangeCodeDf[["exchangeCode"]].copy()
        securityReturnsPdf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityReturns(exchangeCodeDf['exchangeCode'], self.ctx)
        # securityPricesPdf:pd.DataFrame = self.sppMLTrainingDao.loadSecurityPrices(exchangeCodeDf['exchangeCode'], self.ctx)
        # securityVolumePdf = self.extractAndSetupSecurityTradeVolume(securityPricesPdf)


        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for ec in exchangeCodeDf['exchangeCode']:
                future = executor.submit(self.__submitForSppSecurityForecastTask__, forecastIndexReturns
                                         , securityReturnsPdf[securityReturnsPdf['exchangeCode'] == ec]
                                         , interestRatesPdf, inflationRatesPdf)
                futures.append(future)


        for f in futures:
            print(f.result())

        endT = time.time()
        print("SppTrainer - "+self.ctx['pScoreDate']+" - Time taken:" + str(endT - startT) + " secs")

    def setupInterestRates(self, interestRatesPdf:pd.DataFrame) -> pd.DataFrame:

        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        forecastDate = endDate + timedelta(days=self.ctx['forecastDays'])

        # firt get interest rates for training period
        interestRatesReindexPdf = pd.date_range(start=startDate, end=endDate, inclusive="both")
        interestRatesPdf = interestRatesPdf[interestRatesPdf.index.isin(interestRatesReindexPdf)]
        interestRatesPdf = interestRatesPdf.reindex(interestRatesReindexPdf, method='ffill')
        interestRatesPdf.sort_index(inplace=True)

        # assume interest rates remains constant for forecast period
        interestRatesReindexPdf = pd.date_range(start=startDate, end=forecastDate, inclusive="both")
        interestRatesPdf = interestRatesPdf.reindex(interestRatesReindexPdf, method='ffill')
        interestRatesPdf.sort_index(inplace=True)
        return interestRatesPdf

    def setupInflationRates(self, inflationRatesPdf:pd.DataFrame) -> pd.DataFrame:

        startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
        endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
        forecastDate = endDate + timedelta(days=self.ctx['forecastDays'])

        # firt get inflation rates for training period
        inflationRatesReindexPdf = pd.date_range(start=startDate, end=endDate, inclusive="both")
        inflationRatesPdf = inflationRatesPdf[inflationRatesPdf.index.isin(inflationRatesReindexPdf)]
        inflationRatesPdf = inflationRatesPdf.reindex(inflationRatesReindexPdf, method='bfill')
        inflationRatesPdf.sort_index(inplace=True)

        # assume inflation rates remains constant for forecast period
        inflationRatesReindexPdf = pd.date_range(start=startDate, end=forecastDate, inclusive="both")
        inflationRatesPdf = inflationRatesPdf.reindex(inflationRatesReindexPdf, method='ffill')
        inflationRatesPdf.sort_index(inplace=True)
        return inflationRatesPdf

    # def extractAndSetupSecurityTradeVolume(self, securityPricesPdf: pd.DataFrame) -> pd.DataFrame:
    #
    #     securityPricesPdf['datetime'] = pd.to_datetime(securityPricesPdf['trainingDate'])
    #     securityPricesPdf.set_index('datetime', drop=True, inplace=True)
    #     securityPricesPdf.sort_index(inplace=True)
    #     securityVolumePdf = securityPricesPdf[['trainingDate', 'exchangeCode', 'volume']].copy()
    #     startDate = datetime.strptime(self.ctx['trainingStartDate'], '%Y-%m-%d')
    #     endDate = datetime.strptime(self.ctx['trainingEndDate'], '%Y-%m-%d')
    #     securityVolumeReindexPdf = pd.date_range(start=startDate, end=endDate, inclusive="both")
    #     securityVolumePdf = securityVolumePdf.reindex(securityVolumeReindexPdf, fill_value=0)
    #     securityVolumePdf.sort_index(inplace=True)
    #     return securityVolumePdf