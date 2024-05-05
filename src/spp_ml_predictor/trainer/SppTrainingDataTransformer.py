import pandas as pd
from ..trainer import SppCandleStick
import numpy as np
from ..util import util

class SppTrainingDataTransformer:
    def __init__(self, trainingData:dict, trainingDataCtx:dict):
        self.trainingData = trainingData
        self.trainingDataCtx = trainingDataCtx

    def __transformIndexData__(self):
        indexLevelsPdf:pd.DataFrame = self.trainingData['indexLevelsPdf']
        indexLevelsPdf['datetime'] = pd.to_datetime(indexLevelsPdf['date'])
        indexLevelsPdf.set_index("datetime", inplace=True, drop=True)
        indexLevelsPdf.sort_index(inplace=True)
        self.trainingData['indexLevelsPdf'] = indexLevelsPdf

    def __transformSecurityData__(self):
        securityPricesPdf:pd.DataFrame = self.trainingData['securityPricesPdf']
        securityPricesPdf['datetime'] = pd.to_datetime(securityPricesPdf['date'])
        securityPricesPdf.set_index("datetime", inplace=True, drop=True)
        securityPricesPdf.sort_index(inplace=True)
        self.trainingData['securityPricesPdf'] = securityPricesPdf

    def __candlestick_movement__(self, row):
        return util.candlestick_movement(row['open'], row['close'])

    def __determineIndexCandlestickPatterns__(self):
        indexLevelsPdf: pd.DataFrame = self.trainingData['indexLevelsPdf']
        indexLevelsPdf['candlestickMovement'] = indexLevelsPdf.apply(self.__candlestick_movement__, axis=1)

    def __determineSecurityCandlestickPatterns__(self):
        securityPricesPdf: pd.DataFrame = self.trainingData['securityPricesPdf']
        securityPricesPdf['candlestickMovement'] = securityPricesPdf.apply(self.__candlestick_movement__, axis=1)


    def transform(self):
        self.__transformIndexData__()
        self.__determineIndexCandlestickPatterns__()
        self.__transformSecurityData__()
        self.__determineSecurityCandlestickPatterns__()
        return self.trainingData