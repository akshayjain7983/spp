import pyspark.sql as ps
import pyspark.sql.functions as psf
import pandas as pd
from ..dao import QueryFiles
from ..dao import QueryHolder
import json
import sys

class SppMLTrainingDao:

    def __init__(self, spark:ps.SparkSession):
        self.spark = spark

    def loadSecurityExchangeCodes(self, ctx) -> ps.DataFrame:
        exchange = ctx['exchange']
        exchange_code_global = ctx['exchangeCode']
        security_codes_collection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL,"loadSecurityExchangeCodesCollectionName")
        security_codes_mql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadSecurityExchangeCodesMql")
        security_codes_mql = security_codes_mql.format(exchange, ((', "exchangeCode":"'+exchange_code_global+'"') if exchange_code_global else ''))
        security_codes_mql_dict = eval(security_codes_mql)
        return self.spark.read.format("mongodb") \
                        .option("database", "spp") \
                        .option("collection", security_codes_collection) \
                        .option("aggregation.pipeline", security_codes_mql_dict) \
                        .load()

    def loadSecurityPrices(self, exchangeCodesList, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr + '"' + ec.asDict()['exchangeCode'] + '",'

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')
        securityPricesCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL,"loadSecurityPricesCollectionName");
        securityPricesMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadSecurityPricesMql");
        securityPricesMql = securityPricesMql.format(exchange, trainingStartDate, trainingEndDate, exchangeCodesInStr);
        securityPricesMql_dict = json.loads(securityPricesMql)
        return self.spark.read.format("mongodb") \
                        .option("database", "spp") \
                        .option("collection", securityPricesCollection) \
                        .option("aggregation.pipeline", securityPricesMql_dict) \
                        .load()

    def loadIndexLevels(self, ctx) -> ps.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']
        indexLevelsCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL,"loadIndexLevelsCollectionName");
        indexLevelsMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadIndexLevelsMql");
        indexLevelsMql = indexLevelsMql.format(exchange, index, trainingStartDate, trainingEndDate);
        indexLevelsMql_dict = json.loads(indexLevelsMql)
        return self.spark.read.format("mongodb") \
                        .option("database", "spp") \
                        .option("collection", indexLevelsCollection) \
                        .option("aggregation.pipeline", indexLevelsMql_dict) \
                        .load()

    def loadSecurityTrainingPScore(self, exchangeCodesList, ctx) -> pd.DataFrame:

        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr+'"'+ec+'",'

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')

        securityTrainingPScoreCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL,"loadSecurityTrainingPScoreCollectionName");
        securityTrainingPScoreMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL,"loadSecurityTrainingPScoreMql");
        securityTrainingPScoreMql = securityTrainingPScoreMql.format(exchange, index, trainingStartDate, trainingEndDate, exchangeCodesInStr);
        securityTrainingPScoreMql_dict = json.loads(securityTrainingPScoreMql)
        
        results = self.mongoClient['spp'][securityTrainingPScoreCollection].aggregate(securityTrainingPScoreMql_dict)
        return pd.DataFrame(list(results))

    def loadInterestRates(self, institution, rateType) -> ps.DataFrame:

        interestRatesCollectionName = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadInterestRatesCollectionName");
        interestRatesMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadInterestRatesMql")
        interestRatesMql = interestRatesMql.format(institution, rateType)
        interestRatesMql_dict = json.loads(interestRatesMql)
        interestRates:ps.DataFrame = self.spark.read.format("mongodb") \
                                                        .option("database", "spp") \
                                                        .option("collection", interestRatesCollectionName) \
                                                        .option("aggregation.pipeline", interestRatesMql_dict) \
                                                        .load()
                                    
                          
        interestRates = interestRates.withColumn('date', psf.to_date(interestRates['date'], 'dd-MM-yyyy')).sort('date')
        return interestRates.select('rateType', 'date', 'rate')

    def loadInflationRates(self, institution, rateType) -> ps.DataFrame:

        inflationRatesCollectionName = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadInflationRatesCollectionName");
        inflationRatesMql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "loadInflationtRatesMql")
        inflationRatesMql = inflationRatesMql.format(institution, rateType)
        inflationRatesMql_dict = json.loads(inflationRatesMql)
        inflationRates:ps.DataFrame = self.spark.read.format("mongodb") \
                                                        .option("database", "spp") \
                                                        .option("collection", inflationRatesCollectionName) \
                                                        .option("aggregation.pipeline", inflationRatesMql_dict) \
                                                        .load()
                                                        
        inflationRates = inflationRates.withColumn('date', psf.to_date(inflationRates['date'], 'yyyy-MM-dd')).sort('date')
        return inflationRates.select('rateType', 'date', 'rate')


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

        forecastPScoreCollection = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SPARK_MQL, "saveForecastPScoreCollectionName");
        self.mongoClient['spp'][forecastPScoreCollection].update_many(forecastPScoreForUpsert, forecastPScoreForSave, upsert=True)

    def loadTrainingData(self, ctx:dict):

        interestRatesPdf: ps.DataFrame = self.loadInterestRates("Reserve Bank of India", "repo")
        inflationRatesPdf: ps.DataFrame = self.loadInflationRates("Reserve Bank of India","CPI - YoY - General")
        indexLevelsPdf: ps.DataFrame = self.loadIndexLevels(ctx)
        exchangeCodePdf: ps.DataFrame = self.loadSecurityExchangeCodes(ctx)
        exchangeCodePdf = exchangeCodePdf.select("exchangeCode").toLocalIterator(True)
        securityPricesPdf: ps.DataFrame = self.loadSecurityPrices(exchangeCodePdf, ctx)

        return {
            'interestRatesPdf': interestRatesPdf
            , 'inflationRatesPdf': inflationRatesPdf
            , 'exchangeCodePdf': exchangeCodePdf
            , 'indexLevelsPdf': indexLevelsPdf
            , 'securityPricesPdf': securityPricesPdf
        }