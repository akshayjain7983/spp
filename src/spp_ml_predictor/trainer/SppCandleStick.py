import numpy as np

class SppCandleStick:
    def __init__(self, opn:float, high:float, low:float, close:float):
        self.open = opn
        self.high = high
        self.low = low
        self.close = close

        if(self.close >= self.open):
            self.color = 'G'
            self.topWickLen = self.high - self.close
            self.bottomWickLen = self.open - self.low
            self.body = self.close - self.open
        else:
            self.color = 'R'
            self.topWickLen = self.high - self.open
            self.bottomWickLen = self.close - self.low
            self.body = self.open - self.close

        self.movement = self.close / self.open - 1
        self.realBodyChange = self.close / self.open
        self.fullBodyChange = self.high / self.low

    def getPattern(self, candleSticks:[]):

        c3:SppCandleStick = candleSticks[0]
        c2:SppCandleStick = candleSticks[1]
        c1:SppCandleStick = candleSticks[2]

        if (self.__isFallingThree__(candleSticks)):
            return 'fallingThree'

        if (self.__isRisingThree__(candleSticks)):
            return 'risingThree'

        if(self.__isHammer__(candleSticks)):
            return 'hammer'

        if (self.__isInverseHammer__(candleSticks)):
            return 'inverseHammer'

        if (self.__isBullishEngulfing__(candleSticks)):
            return 'bullishEngulfing'

        if (self.__isPiercingLine__(candleSticks)):
            return 'piercingLine'

        if (self.__isMorningStar__(candleSticks)):
            return 'morningStar'

        if (self.__isThreeWhiteSoldiers__(candleSticks)):
            return 'threeWhiteSoldiers'

        if (self.__isHangingMan__(candleSticks)):
            return 'hangingMan'

        if (self.__isShootingStar__(candleSticks)):
            return 'shootingStar'

        if (self.__isBearishEngulfing__(candleSticks)):
            return 'bearishEngulfing'

        if (self.__isEveningStar__(candleSticks)):
            return 'eveningStar'

        if (self.__isThreeBlackCrows__(candleSticks)):
            return 'threeBlackCrows'

        if (self.__isDarkCloudCover__(candleSticks)):
            return 'darkCloudCover'

        return None

    def __colorCount__(self, candleSticks:[], color:str):
        c4: SppCandleStick = candleSticks[0]
        c3: SppCandleStick = candleSticks[1]
        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]

        colorCount = 0
        colorCount += 1 if c4.color == color else 0
        colorCount += 1 if c3.color == color else 0
        colorCount += 1 if c2.color == color else 0
        colorCount += 1 if c1.color == color else 0

        return colorCount

#Continuation trends
    def __isFallingThree__(self, candleSticks:[]):

        c4: SppCandleStick = candleSticks[0]
        c3: SppCandleStick = candleSticks[1]
        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]

        return (c4.color == 'R'
                and c4.movement > 0.05
                and c3.color == 'G'
                and c3.close <= c4.open
                and c3.high <= c4.high
                and c3.movement < 0.03
                and c2.color == 'G'
                and c2.open >= c3.open
                and c2.close >= c3.close
                and c2.close <= c4.open
                and c2.high <= c4.high
                and c2.movement < 0.03
                and c1.color == 'G'
                and c1.open >= c2.open
                and c1.close >= c2.close
                and c1.close <= c4.open
                and c1.high <= c4.high
                and c1.movement < 0.03
                and self.color == 'R'
                and self.close <= c3.open
                and self.low <= c3.low
                and self.movement > 0.05)

    def __isRisingThree__(self, candleSticks:[]):

        c4: SppCandleStick = candleSticks[0]
        c3: SppCandleStick = candleSticks[1]
        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]

        return (c4.color == 'G'
                and c4.movement > 0.05
                and c3.color == 'R'
                and c3.close >= c4.open
                and c3.low >= c4.low
                and c3.movement < 0.03
                and c2.color == 'R'
                and c2.open <= c3.open
                and c2.close <= c3.close
                and c2.close >= c4.open
                and c2.low >= c4.low
                and c2.movement < 0.03
                and c1.color == 'R'
                and c1.open <= c2.open
                and c1.close <= c2.close
                and c1.close >= c4.open
                and c1.low >= c4.low
                and c1.movement < 0.03
                and self.color == 'G'
                and self.close >= c3.open
                and self.high >= c3.high
                and self.movement > 0.05)

# Bullish trends
    def __isHammer__(self, candleSticks:[]):

        redCount = self.__colorCount__(candleSticks, 'R')

        return (self.color == 'G'
                and self.topWickLen < self.bottomWickLen
                and self.topWickLen < self.body
                and self.body < self.bottomWickLen
                and redCount >= 2)

    def __isInverseHammer__(self, candleSticks:[]):

        redCount = self.__colorCount__(candleSticks, 'R')

        return (self.color == 'G'
                and self.topWickLen > self.bottomWickLen
                and self.bottomWickLen < self.body
                and self.body < self.topWickLen
                and redCount >= 2)

    def __isBullishEngulfing__(self, candleSticks:[]):

        c1: SppCandleStick = candleSticks[3]
        redCount = self.__colorCount__(candleSticks, 'R')

        return (self.color == 'G'
                and c1.color == 'R'
                and self.open < c1.open
                and self.close > c1.open
                and redCount >= 2)

    def __isPiercingLine__(self, candleSticks:[]):

        c1: SppCandleStick = candleSticks[3]
        redCount = self.__colorCount__(candleSticks, 'R')

        return (self.color == 'G'
                and c1.color == 'R'
                and np.abs(self.movement) > 0.05
                and np.abs(c1.movement) > 0.05
                and self.open < c1.close
                and self.close > ((c1.open + c1.close) / 2))

    def __isMorningStar__(self, candleSticks:[]):

        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]
        redCount = self.__colorCount__(candleSticks, 'R')

        return (self.color == 'G'
                and c1.color == 'R'
                and c2.color == 'R'
                and np.abs(self.movement) > 0.05
                and np.abs(c2.movement) > 0.05
                and np.abs(c1.movement) < 0.01
                and redCount >= 2)

    def __isThreeWhiteSoldiers__(self, candleSticks:[]):

        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]

        return (self.color == 'G'
                and c1.color == 'G'
                and c2.color == 'G'
                and np.abs(self.movement) > 0.06
                and np.abs(c2.movement) > 0.04
                and np.abs(c1.movement) > 0.02
                and c2.open >= (c1.close * 0.99)
                and self.open >= (c2.close * 0.99))

# Bearish trends
    def __isHangingMan__(self, candleSticks:[]):

        greenCount = self.__colorCount__(candleSticks, 'G')

        return (self.color == 'R'
                and self.topWickLen < self.bottomWickLen
                and self.topWickLen < self.body
                and self.body < self.bottomWickLen
                and greenCount >= 2)

    def __isShootingStar__(self, candleSticks:[]):

        greenCount = self.__colorCount__(candleSticks, 'G')

        return (self.color == 'R'
                and self.topWickLen > self.bottomWickLen
                and self.bottomWickLen < self.body
                and self.body < self.topWickLen
                and greenCount >= 2)

    def __isBearishEngulfing__(self, candleSticks:[]):

        c1: SppCandleStick = candleSticks[3]
        greenCount = self.__colorCount__(candleSticks, 'G')

        return (self.color == 'R'
                and c1.color == 'G'
                and self.open > c1.open
                and self.close < c1.open
                and greenCount >= 2)

    def __isEveningStar__(self, candleSticks:[]):

        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]
        greenCount = self.__colorCount__(candleSticks, 'G')

        return (self.color == 'R'
                and c1.color == 'G'
                and c2.color == 'G'
                and np.abs(self.movement) > 0.05
                and np.abs(c2.movement) > 0.05
                and np.abs(c1.movement) < 0.01
                and greenCount >= 2)

    def __isThreeBlackCrows__(self, candleSticks:[]):

        c2: SppCandleStick = candleSticks[2]
        c1: SppCandleStick = candleSticks[3]

        return (self.color == 'R'
                and c1.color == 'R'
                and c2.color == 'R'
                and np.abs(self.movement) > 0.06
                and np.abs(c2.movement) > 0.04
                and np.abs(c1.movement) > 0.02
                and c2.open <= (c1.close * 0.99)
                and self.open <= (c2.close * 0.99))

    def __isDarkCloudCover__(self, candleSticks:[]):

        c1: SppCandleStick = candleSticks[3]
        greenCount = self.__colorCount__(candleSticks, 'G')

        return (self.color == 'R'
                and c1.color == 'G'
                and np.abs(self.movement) > 0.05
                and np.abs(c1.movement) > 0.05
                and self.open > c1.high
                and self.close < ((c1.open + c1.close) / 2))