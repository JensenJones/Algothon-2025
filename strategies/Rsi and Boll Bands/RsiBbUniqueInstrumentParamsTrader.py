from typing import Optional

import numpy as np


class BollingerBandsCalculator:
    """
    Middle band is the standard moving average (SMA) of the instrument price over D days.
    Upper band is the SMA + standard deviation of closing price over D days
    Lower Band is the SMA - standard deviation of closing price over D days
    """

    def __init__(self, prices: np.ndarray, windowSize: int):
        super().__init__()

        self.prices = prices
        self.windowSize = windowSize
        self.windowSum = None
        self.windowElements = None
        self.sma = None
        self.queueIndex = 0

        self.calcSma()
        self.bollingerBands = np.zeros((50, 2)) # [:, 1] = upperBand, [:, 0] = lowerBand
        self.calcBollingerBands()


    def updateWithNewDay(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.updateSma(newDayPrices)
        self.calcBollingerBands()

    def calcSma(self):
        self.windowElements = self.prices[:, -self.windowSize:].copy()
        self.windowSum = np.sum(self.windowElements, axis = 1)
        self.sma = self.windowSum / self.windowSize

    def updateSma(self, newDayPrices: np.ndarray):
        self.windowSum -= self.windowElements[:, self.queueIndex]
        self.windowElements[:, self.queueIndex] = newDayPrices
        self.windowSum += newDayPrices

        self.queueIndex = (self.queueIndex + 1) % self.windowSize

        self.sma = self.windowSum / self.windowSize

    def calcBollingerBands(self):
        stDev = np.std(self.windowElements, axis = 1)
        self.bollingerBands[:, 1] = self.sma + 2 * stDev
        self.bollingerBands[:, 0] = self.sma - 2 * stDev

    def getUpperBands(self):
        return self.bollingerBands[:, 1]

    def getLowerBands(self):
        return self.bollingerBands[:, 0]

    def getSma(self):
        return self.sma

class RsiCalculator:
    def __init__(self, prices, windowSize):
        self.prices = prices
        self.windowSize = windowSize

        self.avgGain = None
        self.avgLoss = None
        self.rsi = None

        self.initializeAverages()

    def updateWithNewDay(self, newDayPrices):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])

        if self.prices.shape[1] < self.windowSize + 1:
            return None

        if self.avgGain is None or self.avgLoss is None:
            self.initializeAverages()
            return self.rsi

        self.calculateRsi(newDayPrices)
        return self.rsi

    def initializeAverages(self):
        if self.prices.shape[1] < self.windowSize + 1:
            return

        delta = np.diff(self.prices[:, -self.windowSize - 1:], axis=1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        self.avgGain = np.mean(gain, axis=1)
        self.avgLoss = np.mean(loss, axis=1)

        rs = self.avgGain / (self.avgLoss + 1e-10)
        self.rsi = 100 - (100 / (1 + rs))

    def calculateRsi(self, newDayPrices):
        prevDayPrices = self.prices[:, -2]
        delta = newDayPrices - prevDayPrices
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        self.avgGain = (self.avgGain * (self.windowSize - 1) + gain) / self.windowSize
        self.avgLoss = (self.avgLoss * (self.windowSize - 1) + loss) / self.windowSize

        rs = self.avgGain / (self.avgLoss + 1e-10)
        self.rsi = 100 - (100 / (1 + rs))

    def getRsi(self):
        return self.rsi

class RsiBollingerBandsTrader:
    def __init__(self, logFilePath, prices, bollingerBandsCalculator, rsiCalculator, purchaseAlpha: int, rsiLongThreshold=30, rsiShortThreshold=70):
        super().__init__()

        # Always ensure prices is shape (1, n_days)
        self.prices = prices
        self.logFilePath = logFilePath
        self.bbc = bollingerBandsCalculator
        self.rsiC = rsiCalculator
        self.purchaseAlpha = purchaseAlpha
        self.rsiLongThreshold = rsiLongThreshold
        self.rsiShortThreshold = rsiShortThreshold

        if logFilePath is not None:
            open(logFilePath, "wb").close()
            self.logAll()

    def updatePosition(self, newDayPrice, position):
        # newDayPrice: float or shape (1,)
        # position: float or shape (1,)

        newDayPrice = np.atleast_1d(newDayPrice)
        position = np.atleast_1d(position)
        self.prices = np.hstack([self.prices, newDayPrice.reshape(1, 1)])

        self.bbc.updateWithNewDay(newDayPrice)
        self.rsiC.updateWithNewDay(newDayPrice)

        if self.prices.shape[1] < max(self.bbc.windowSize, self.rsiC.windowSize):
            return position

        upperBand = self.bbc.getUpperBands()
        lowerBand = self.bbc.getLowerBands()
        rsi = self.rsiC.getRsi()

        if rsi is None:
            print("rsi is none in update position\n")
            return position

        # Simple, not mask: 1D instrument
        if newDayPrice[0] < lowerBand[0] and rsi[0] < self.rsiLongThreshold:
            position[0] += (self.purchaseAlpha * (self.rsiLongThreshold - rsi[0]) / self.rsiLongThreshold)
        if newDayPrice[0] > upperBand[0] and rsi[0] > self.rsiShortThreshold:
            position[0] -= (self.purchaseAlpha * (rsi[0] - self.rsiShortThreshold) / (100 - self.rsiShortThreshold))

        self.logAll()
        return position

    def logAll(self):
        if self.logFilePath is None:
            return

        with open(self.logFilePath, "ab") as logFile:
            lowerBand = self.bbc.getLowerBands()
            upperBand = self.bbc.getUpperBands()
            sma = self.bbc.getSma()
            rsi = self.rsiC.getRsi()
            dayPrice = self.prices[:, -1]  # shape (1,)

            if lowerBand is None or upperBand is None or sma is None or rsi is None or dayPrice is None:
                return

            # All are 1-element arrays
            log = np.hstack([
                lowerBand.reshape(-1, 1),
                upperBand.reshape(-1, 1),
                sma.reshape(-1, 1),
                rsi.reshape(-1, 1),
                dayPrice.reshape(-1, 1)
            ])  # shape (1, 5)

            np.save(logFile, log)

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

#TODO calculate money position
#TODO calculate profit per instrument

instrumentCount = 50
POSITION_LIMIT = 10_000

windowList: list[Optional[int]] = [None] * instrumentCount
bbcList: list[Optional[BollingerBandsCalculator]] = [None] * instrumentCount
rsiList: list[Optional[RsiCalculator]] = [None] * instrumentCount
traderList: list[Optional[RsiBollingerBandsTrader]] = [None] * instrumentCount

toInit = set(range(50))

currentPos = np.zeros(instrumentCount)

best_params = np.load('./strategies/Rsi and Boll Bands/best_params_per_instrument.npy')  # shape (50, 4)

def getMyPosition(prcSoFar):
    global instrumentCount, currentPos, bbcList, rsiList, traderList, best_params

    day = prcSoFar.shape[1]

    for index in list(toInit):
        window = int(best_params[index, 0])

        if window > day:
            continue

        toInit.remove(index)

        quantityAlpha = best_params[index, 1]
        long_thr = best_params[index, 2]
        short_thr = best_params[index, 3]

        bbc = BollingerBandsCalculator(prcSoFar[index:index+1, :], window)
        rsi = RsiCalculator(prcSoFar[index:index+1, :], window)
        trader = RsiBollingerBandsTrader(None, prcSoFar[index:index+1, :], bbc, rsi, quantityAlpha, long_thr, short_thr)

        bbcList[index] = bbc
        rsiList[index] = rsi
        traderList[index] = trader

    for index in range(instrumentCount):
        if int(best_params[index, 0]) >= day or index in toInit:
            continue

        # Update position for this instrument only
        cur_price = prcSoFar[index, -1].reshape(1,)
        cur_pos = np.array([currentPos[index]])
        currentPos[index] = traderList[index].updatePosition(cur_price, cur_pos)[0]

    # Clip to position limits
    posLimits = np.floor(POSITION_LIMIT / prcSoFar[:, -1]).astype(int)
    currentPos = np.clip(currentPos, -posLimits, posLimits)
    return currentPos
