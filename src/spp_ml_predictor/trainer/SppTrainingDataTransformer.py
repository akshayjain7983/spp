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
        securityPricesPdf['datetime'] = pd.to_datetime(securityPricesPdf['tradingDate'])
        securityPricesPdf.set_index("datetime", inplace=True, drop=True)
        securityPricesPdf.sort_index(inplace=True)
        self.trainingData['securityPricesPdf'] = securityPricesPdf

    def __initCandleStick__(self, row):
        return SppCandleStick.SppCandleStick(row['open'], row['high'], row['low'], row['close'])

    def __determineCandlestickPatterns__(self):
        securityPricesPdf: pd.DataFrame = self.trainingData['securityPricesPdf']
        securityPricesPdf['candlestick'] = securityPricesPdf.apply(self.__initCandleStick__, axis=1)

        for i in range(len(securityPricesPdf)):
            c4 = securityPricesPdf.iloc[i - 4]['candlestick']
            c3 = securityPricesPdf.iloc[i - 3]['candlestick']
            c2 = securityPricesPdf.iloc[i - 2]['candlestick']
            c1 = securityPricesPdf.iloc[i - 1]['candlestick']
            c0 = securityPricesPdf.iloc[i]['candlestick']
            pattern = c0.getPattern([c4, c3, c2, c1])
            patternWeight = 1
            if (pattern in ['hammer', 'inverseHammer', 'bullishEngulfing', 'piercingLine', 'morningStar']):
                patternWeight = 2
            elif (pattern in ['threeWhiteSoldiers']):
                patternWeight = 4
            elif(pattern in ['hangingMan', 'shootingStar', 'bearishEngulfing', 'eveningStar', 'darkCloudCover']):
                patternWeight = 2
            elif (pattern in ['threeBlackCrows']):
                patternWeight = 4

            candleStickRealBodyChange = c0.movement
            # candleStickRealBodyChange = np.log(c0.realBodyChange) * patternWeight
            securityPricesPdf.loc[securityPricesPdf.index[i], 'candleStickRealBodyChange'] = candleStickRealBodyChange
            # candleStickFullBodyChange = np.log(c0.fullBodyChange)
            # securityPricesPdf.loc[securityPricesPdf.index[i], 'candleStickFullBodyChange'] = candleStickFullBodyChange


    def transform(self):
        self.__transformIndexData__()
        self.__transformSecurityData__()
        self.__determineCandlestickPatterns__()
        return self.trainingData