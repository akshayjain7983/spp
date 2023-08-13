from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from ..dao import QueryFiles
from ..dao import QueryHolder

class SppMLTrainingDao:

    def __init__(self, spark:SparkSession):
        self.spark = spark

    def loadSecurityExchangeCodes(self, ctx) -> DataFrame:
        exchange = ctx['exchange']
        exchange_code_global = ctx['exchangeCode']
        security_codes_collection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityExchangeCodesCollectionName")
        security_codes_mql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL, "loadSecurityExchangeCodesMql")
        security_codes_mql = security_codes_mql.format(exchange, ((', exchangeCode:"'+exchange_code_global+'"') if exchange_code_global else ''))

        return (self.spark.read.format("mongodb")
                .option("spark.mongodb.read.collection", security_codes_collection)
                .option("spark.mongodb.read.aggregation.pipeline", security_codes_mql)
                .load())


    def loadSecurityTrainingPScore(self, exchangeCodesList, ctx) -> DataFrame:

        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr+'"'+ec['exchangeCode']+'",'

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')

        securityTrainingPScoreCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityTrainingPScoreCollectionName");
        securityTrainingPScoreMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_MQL,"loadSecurityTrainingPScoreMql");
        securityTrainingPScoreMql = securityTrainingPScoreMql.format(exchange, index, trainingStartDate, trainingEndDate, exchangeCodesInStr);
        return (self.spark.read.format("mongodb")
                .option("spark.mongodb.read.collection", securityTrainingPScoreCollection)
                .option("spark.mongodb.read.aggregation.pipeline", securityTrainingPScoreMql)
                .load())
