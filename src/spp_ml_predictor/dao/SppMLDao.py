import pandas as pd
from ..dao import QueryFiles
from ..dao import QueryHolder
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy import insert

class SppMLDao:

    def __init__(self, config:dict):
        self.engine = create_engine(config['datasource.url'])

    def loadSecurityExchangeCodes(self, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        exchange_code_global = ctx['exchangeCode']
        security_codes_sql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadSecurityExchangeCodes")
        security_codes_sql = security_codes_sql.format((("AND exchange_code IN('"+exchange_code_global+"')") if exchange_code_global else ''))
        return pd.read_sql_query(text(security_codes_sql), self.engine, params={'exchange': exchange})

    def loadSecurityPrices(self, exchangeCodesList, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        exchangeCodesInStr = ""
        for ec in exchangeCodesList:
            exchangeCodesInStr = exchangeCodesInStr + "'" + ec + "',"

        exchangeCodesInStr = exchangeCodesInStr.removesuffix(',')
        securityPricesSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadSecurityPrices")
        securityPricesSql = securityPricesSql.format(exchangeCodesInStr)
        return pd.read_sql_query(text(securityPricesSql)
                                 , self.engine
                                 , params={'exchange':exchange
                                            , 'trainingStartDate':trainingStartDate
                                            , 'trainingEndDate':trainingEndDate})

    def loadIndexLevels(self, ctx) -> pd.DataFrame:
        exchange = ctx['exchange']
        trainingStartDate = ctx['trainingStartDate']
        trainingEndDate = ctx['trainingEndDate']
        index = ctx['index']

        indexLevelsSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadIndexLevels");
        return pd.read_sql_query(text(indexLevelsSql)
                                 , self.engine
                                 , params={'exchange': exchange
                                            , 'index': index
                                            , 'trainingStartDate': trainingStartDate
                                            , 'trainingEndDate': trainingEndDate})

    def loadInterestRates(self, institution, rateType):

        interestRatesSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadInterestRates")
        interestRates = pd.read_sql_query(text(interestRatesSql)
                                            , self.engine
                                            , params={'institution':institution, 'rateType':rateType}
                                            , index_col='date')
        interestRates = interestRates[~interestRates.index.duplicated(keep='first')]
        interestRatesReindexPdf = pd.date_range(start=interestRates.index.min(),end=interestRates.index.max())
        interestRates = interestRates.reindex(interestRatesReindexPdf, method='ffill')
        return interestRates

    def loadInflationRates(self, institution, rateType):

        inflationRatesSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadInflationRates")
        inflationRates = pd.read_sql_query(text(inflationRatesSql)
                                            , self.engine
                                            , params={'institution':institution, 'rateType':rateType}
                                            , index_col='date')
        inflationRates = inflationRates[~inflationRates.index.duplicated(keep='first')]
        inflationRatesReindexPdf = pd.date_range(start=inflationRates.index.min(),end=inflationRates.index.max())
        inflationRates = inflationRates.reindex(inflationRatesReindexPdf, method='ffill')
        return inflationRates

    def loadTrainingData(self, ctx:dict):

        exchangeCodePdf: pd.DataFrame = self.loadSecurityExchangeCodes(ctx)
        exchangeCodePdf = exchangeCodePdf[["exchange_code"]].copy()
        interestRatesPdf: pd.DataFrame = self.loadInterestRates("Reserve Bank of India", "repo")
        inflationRatesPdf: pd.DataFrame = self.loadInflationRates("Reserve Bank of India","CPI - YoY - General")
        indexLevelsPdf: pd.DataFrame = self.loadIndexLevels(ctx)
        securityPricesPdf: pd.DataFrame = self.loadSecurityPrices(exchangeCodePdf['exchange_code'], ctx)

        return {
            'interestRatesPdf': interestRatesPdf
            , 'inflationRatesPdf': inflationRatesPdf
            , 'exchangeCodePdf': exchangeCodePdf
            , 'indexLevelsPdf': indexLevelsPdf
            , 'securityPricesPdf': securityPricesPdf
        }

    def loadHolidays(self, ctx:dict):
        holidaysSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadHolidays")
        return pd.read_sql_query(text(holidaysSql)
                                 , self.engine
                                 , params=ctx
                                 , index_col="date")

    def saveForecastPScore(self, forecastPScore: pd.DataFrame, ctx: dict):

        forecastedPScoreSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "saveForecastedPScore")
        data = []
        for index, row in forecastPScore.iterrows():
            dbRow = {'security_id':row['security_id'], 'index_id':row['index_id'], 'date':row['date'], 'forecast_model_name':row['forecast_model_name'],
                             'forecast_period':row['forecast_period'], 'forecast_date':row['forecast_date'], 'forecasted_index_return':row['forecasted_index_return'],
                             'forecasted_security_return':row['forecasted_security_return'], 'forecasted_p_score':row['forecasted_p_score']}
            data.append(dbRow)

        with self.engine.connect() as conn:
            conn.execute(text(forecastedPScoreSql), data)
            conn.commit()

    def loadForecastPScore(self, ctx:dict) -> pd.DataFrame:
        forecastedPScoreSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadForecastedPScore")
        forecastedPScore = pd.read_sql_query(text(forecastedPScoreSql)
                                                , self.engine
                                                , params=ctx
                                                , index_col='date')
        return forecastedPScore

    def loadActualPScore(self, ctx:dict) -> pd.DataFrame:
        actualPScoreSql = QueryHolder.getQuery(QueryFiles.SPP_STOCK_DATA_SQL, "loadActualPScore")
        actualPScore = pd.read_sql_query(text(actualPScoreSql)
                                                , self.engine
                                                , params=ctx
                                                , index_col='date')
        return actualPScore