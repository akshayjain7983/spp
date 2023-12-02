import pyspark.sql as ps
import pyspark.sql.functions as psf
from ..trainer import SppCandleStick

class SppTrainingDataTransformer:
    def __init__(self, trainingData:dict, trainingDataCtx:dict):
        self.trainingData = trainingData
        self.trainingDataCtx = trainingDataCtx

    def __transformIndexData__(self):
        indexLevelsPdf:ps.DataFrame = self.trainingData['indexLevelsPdf']
        indexLevelsPdf = indexLevelsPdf.withColumn('date', psf.to_date(indexLevelsPdf['date'], 'yyyy-MM-dd')).sort('date')
        self.trainingData['indexLevelsPdf'] = indexLevelsPdf

    def __transformSecurityData__(self):
        securityPricesPdf:ps.DataFrame = self.trainingData['securityPricesPdf']
        securityPricesPdf = securityPricesPdf.withColumn('date', psf.to_date(securityPricesPdf['tradingDate'], 'yyyy-MM-dd')).sort('date')
        self.trainingData['securityPricesPdf'] = securityPricesPdf

    
    def __determineIndexCandlestickPatterns__(self):
        indexLevelsPdf: ps.DataFrame = self.trainingData['indexLevelsPdf']
        udf = psf.udf(lambda row:SppCandleStick.SppCandleStick(row['open'], row['high'], row['low'], row['close']).movementReal)
        indexLevelsPdf = indexLevelsPdf.withColumn('candlestickMovementReal', udf(psf.struct([indexLevelsPdf[x] for x in indexLevelsPdf.columns])))
        self.trainingData['indexLevelsPdf'] = indexLevelsPdf

    def __determineSecurityCandlestickPatterns__(self):
        securityPricesPdf: ps.DataFrame = self.trainingData['securityPricesPdf']
        udf = psf.udf(lambda row:SppCandleStick.SppCandleStick(row['open'], row['high'], row['low'], row['close']).movementReal)
        securityPricesPdf = securityPricesPdf.withColumn('candlestickMovementReal', udf(psf.struct([securityPricesPdf[x] for x in securityPricesPdf.columns])))
        self.trainingData['securityPricesPdf'] = securityPricesPdf
        

    def transform(self):
        self.__transformIndexData__()
        self.__determineIndexCandlestickPatterns__()
        self.__transformSecurityData__()
        self.__determineSecurityCandlestickPatterns__()
        return self.trainingData