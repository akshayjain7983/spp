from pyspark.sql import SparkSession
# from pyspark.sql import DataFrame
import pandas as pd
from ..dao import QueryFiles
from ..dao import QueryHolder
from pymongo import MongoClient
import json
from datetime import datetime

class SppMLTrainingDao:

    def __init__(self):
        self.mongoClient = MongoClient("mongodb://localhost:27017")

    def loadSecurityExchangeCodes(self, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        exchange_code_global = ctx['exchangeCode']
        security_codes_collection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityExchangeCodesCollectionName")
        security_codes_mql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadSecurityExchangeCodesMql")
        security_codes_mql = security_codes_mql.format(exchange, ((', "exchangeCode":"'+exchange_code_global+'"') if exchange_code_global else ''))
        security_codes_mql_dict = eval(security_codes_mql)
        results = self.mongoClient['spp'][security_codes_collection].find(security_codes_mql_dict)
        return pd.DataFrame(list(results))

    def loadSecurityReturns(self, exchangeCodesList, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr + '"' + ec + '",'

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')
        securityReturnsCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityReturnsCollectionName");
        securityReturnsMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadSecurityReturnsMql");
        securityReturnsMql = securityReturnsMql.format(exchange, trainingStartDate, trainingEndDate, exchangeCodesInStr);
        securityReturnsMql_dict = json.loads(securityReturnsMql)
        results = self.mongoClient['spp'][securityReturnsCollection].find(securityReturnsMql_dict)
        return pd.DataFrame(list(results))

    def loadSecurityPrices(self, exchangeCodesList, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr + '"' + ec + '",'

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')
        securityPricesCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityPricesCollectionName");
        securityPricesMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadSecurityPricesMql");
        securityPricesMql = securityPricesMql.format(exchange, trainingStartDate, trainingEndDate, exchangeCodesInStr);
        securityPricesMql_dict = json.loads(securityPricesMql)
        results = self.mongoClient['spp'][securityPricesCollection].find(securityPricesMql_dict)
        return pd.DataFrame(list(results))

    def loadIndexReturns(self, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']

        indexReturnsCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadIndexReturnsCollectionName");
        indexReturnsMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadIndexReturnsMql");
        indexReturnsMql = indexReturnsMql.format(exchange, index, trainingStartDate, trainingEndDate);
        indexReturnsMql_dict = json.loads(indexReturnsMql)
        results = self.mongoClient['spp'][indexReturnsCollection].find(indexReturnsMql_dict)
        return pd.DataFrame(list(results))

    def loadIndexLevels(self, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']

        indexLevelsCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadIndexLevelsCollectionName");
        indexLevelsMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadIndexLevelsMql");
        indexLevelsMql = indexLevelsMql.format(exchange, index, trainingStartDate, trainingEndDate);
        indexLevelsMql_dict = json.loads(indexLevelsMql)
        results = self.mongoClient['spp'][indexLevelsCollection].find(indexLevelsMql_dict)
        return pd.DataFrame(list(results))

    def loadSecurityTrainingPScore(self, exchangeCodesList, ctx) -> pd.DataFrame:

        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr+'"'+ec+'",'

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')

        securityTrainingPScoreCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityTrainingPScoreCollectionName");
        securityTrainingPScoreMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityTrainingPScoreMql");
        securityTrainingPScoreMql = securityTrainingPScoreMql.format(exchange, index, trainingStartDate, trainingEndDate, exchangeCodesInStr);
        securityTrainingPScoreMql_dict = json.loads(securityTrainingPScoreMql)
        results = self.mongoClient['spp'][securityTrainingPScoreCollection].aggregate(securityTrainingPScoreMql_dict)
        return pd.DataFrame(list(results))

    def loadInterestRates(self, institution, rateType):

        interestRatesCollectionName = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadInterestRatesCollectionName");
        interestRatesMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadInterestRatesMql")
        interestRatesMql = interestRatesMql.format(institution, rateType)
        interestRatesMql_dict = json.loads(interestRatesMql)
        results = self.mongoClient['spp'][interestRatesCollectionName].find(interestRatesMql_dict)
        interestRates = pd.DataFrame(list(results))
        interestRates['datetime'] = pd.to_datetime(interestRates['date'], format="%d-%m-%Y")
        interestRates.set_index('datetime', inplace=True, drop=True)
        interestRates.sort_index(inplace=True)
        interestRates.drop_duplicates(subset=['date'], inplace=True)
        interestRatesReindexPdf = pd.date_range(start=interestRates.index.min(),end=interestRates.index.max(), inclusive="both")
        interestRates = interestRates.reindex(interestRatesReindexPdf, method='ffill')
        return interestRates

    def loadInflationRates(self, institution, rateType):

        inflationRatesCollectionName = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadInflationRatesCollectionName");
        inflationRatesMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadInflationtRatesMql")
        inflationRatesMql = inflationRatesMql.format(institution, rateType)
        inflationRatesMql_dict = json.loads(inflationRatesMql)
        results = self.mongoClient['spp'][inflationRatesCollectionName].find(inflationRatesMql_dict)
        inflationRates = pd.DataFrame(list(results))
        inflationRates['datetime'] = pd.to_datetime(inflationRates['date'], format="%Y-%m-%d")
        inflationRates.set_index('datetime', inplace=True, drop=True)
        inflationRates.sort_index(inplace=True)
        inflationRates.drop_duplicates(subset=['date'], inplace=True)
        inflationRatesReindexPdf = pd.date_range(start=inflationRates.index.min(),end=inflationRates.index.max(), inclusive="both")
        inflationRates = inflationRates.reindex(inflationRatesReindexPdf, method='bfill')
        return inflationRates


    def saveForecastPScore(self, forecastPScore:pd.DataFrame, ctx:dict):

        forecastDays = ctx['forecastDays']

        forecastPScoreForUpsert = {
            "exchange": forecastPScore["exchange"][forecastDays[1]],
            "index": forecastPScore["index"][forecastDays[1]],
            "exchangeCode": forecastPScore["exchangeCode"][forecastDays[1]],
            "isin": forecastPScore["isin"][forecastDays[1]],
            "date": forecastPScore["date"][forecastDays[1]],
            "forecast.forecastModel": forecastPScore["forecastModel"][forecastDays[1]]
        }

        forecastPScoreWithPeriods = {}
        forecastPScoreWithPeriods['forecastModel'] = forecastPScore["forecastModel"][forecastDays[1]]
        for d in forecastDays[1:]:
            forecastPScoreWithPeriods[forecastPScore['forecastPeriod'][d]] = {
                "forecastDate": forecastPScore["forecastDate"][d],
                "forecastedIndexReturn": forecastPScore["forecastedIndexReturn"][d],
                "forecastedSecurityReturn": forecastPScore["forecastedSecurityReturn"][d],
                "forecastedPScore": forecastPScore["forecastedPScore"][d],
            }

        forecastPScoreForSave = {
            "$setOnInsert":{
                "exchange": forecastPScore["exchange"][forecastDays[1]],
                "index": forecastPScore["index"][forecastDays[1]],
                "exchangeCode": forecastPScore["exchangeCode"][forecastDays[1]],
                "isin": forecastPScore["isin"][forecastDays[1]],
                "date": forecastPScore["date"][forecastDays[1]]
            },
            "$set":{
                "lastUpdatedTimestamp": forecastPScore["lastUpdatedTimestamp"][forecastDays[1]],
                "forecast":forecastPScoreWithPeriods
            }

        }

        forecastPScoreCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "saveForecastPScoreCollectionName");
        self.mongoClient['spp'][forecastPScoreCollection].update_many(forecastPScoreForUpsert, forecastPScoreForSave, upsert=True)

    def loadTrainingData(self, ctx:dict):

        interestRatesPdf: pd.DataFrame = self.loadInterestRates("Reserve Bank of India", "repo")
        inflationRatesPdf: pd.DataFrame = self.loadInflationRates("Reserve Bank of India","CPI - YoY - General")
        indexLevelsPdf: pd.DataFrame = self.loadIndexLevels(ctx)
        exchangeCodePdf: pd.DataFrame = self.loadSecurityExchangeCodes(ctx)
        exchangeCodePdf = exchangeCodePdf[["exchangeCode"]].copy()
        securityPricesPdf: pd.DataFrame = self.loadSecurityPrices(exchangeCodePdf['exchangeCode'], ctx)

        return {
            'interestRatesPdf': interestRatesPdf
            , 'inflationRatesPdf': inflationRatesPdf
            , 'exchangeCodePdf': exchangeCodePdf
            , 'indexLevelsPdf': indexLevelsPdf
            , 'securityPricesPdf': securityPricesPdf
        }