from pyspark.sql import SparkSession
# from pyspark.sql import DataFrame
import pandas as pd
from ..dao import QueryFiles
from ..dao import QueryHolder
from pymongo import MongoClient
import json

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

    def saveForecastPScore(self, forecastPScore):
        forecastPScoreCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "saveForecastPScoreCollectionName");
        forecastPScore.write.format("mongodb").mode("append").option("collection", forecastPScoreCollection).save()