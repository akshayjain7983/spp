import pandas as pd
from ..trainer import SppCandleStick
import numpy as np

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

    def __initCandleStick__(self, row):
        return SppCandleStick.SppCandleStick(row['open'], row['high'], row['low'], row['close'])

    def __determineIndexCandlestickPatterns__(self):
        indexLevelsPdf: pd.DataFrame = self.trainingData['indexLevelsPdf']
        indexLevelsPdf['candlestick'] = indexLevelsPdf.apply(self.__initCandleStick__, axis=1)

        for i in range(len(indexLevelsPdf)):
            c0 = indexLevelsPdf.iloc[i]['candlestick']
            candleStickRealBodyChange = c0.movement
            indexLevelsPdf.loc[indexLevelsPdf.index[i], 'candleStickRealBodyChange'] = candleStickRealBodyChange

    def __determineSecurityCandlestickPatterns__(self):
        securityPricesPdf: pd.DataFrame = self.trainingData['securityPricesPdf']
        securityPricesPdf['candlestick'] = securityPricesPdf.apply(self.__initCandleStick__, axis=1)

        for i in range(len(securityPricesPdf)):
            c0 = securityPricesPdf.iloc[i]['candlestick']
            candleStickRealBodyChange = c0.movement
            securityPricesPdf.loc[securityPricesPdf.index[i], 'candleStickRealBodyChange'] = candleStickRealBodyChange


    def transform(self):
        self.__transformIndexData__()
        self.__determineIndexCandlestickPatterns__()
        self.__transformSecurityData__()
        self.__determineSecurityCandlestickPatterns__()
        return self.trainingData