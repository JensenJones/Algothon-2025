import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

#TODO calculate money position
#TODO calculate profit per instrument

instrumentCount = 50
currentPos = np.zeros(instrumentCount)
bbc = None
rsi = None
trader = None
WINDOW_SIZE = 33
POSITION_LIMIT = 10_000

def getMyPosition(prcSoFar):
    global currentPos, instrumentCount, bbc, rsi, trader, WINDOW_SIZE

    day = prcSoFar.shape[1]

    if day < WINDOW_SIZE:
        return currentPos

    if trader is None:
        filePath = f"./strategies/RsiBollBandsLog_WindowSize={WINDOW_SIZE}.npy"
        bbc = BollingerBandsCalculator(prcSoFar, WINDOW_SIZE)
        rsi = RsiCalculator(prcSoFar, WINDOW_SIZE)
        trader = RsiBollingerBandsTrader(filePath, prcSoFar, bbc, rsi, 162, 50, 90)

    currentPos = trader.updatePosition(prcSoFar[:, -1], currentPos)

    return currentPos

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
    def __init__(self, logFilePath, prices, bollingerBandsCalculator, rsiCalculator, purchaseAlpha: int, rsiLongThreshold = 30, rsiShortThreshold = 70):
        super().__init__()

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

    def updatePosition(self, newDayPrices: np.ndarray, position: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.bbc.updateWithNewDay(newDayPrices)
        self.rsiC.updateWithNewDay(newDayPrices)

        upperBands = self.bbc.getUpperBands()
        lowerBands = self.bbc.getLowerBands()
        rsi = self.rsiC.getRsi()

        if rsi is None:
            print("rsi is none in update position\n")

        longMask =  (newDayPrices < lowerBands) & (rsi < self.rsiLongThreshold)
        shortMask = (newDayPrices > upperBands) & (rsi > self.rsiShortThreshold)

        # Confidence scaling of position change (the further past the threshold, the more we change position)
        position[longMask] += (self.purchaseAlpha * (self.rsiLongThreshold - rsi[longMask])
                               / self.rsiLongThreshold)
        position[shortMask] -= ((self.purchaseAlpha * (rsi[shortMask] - self.rsiShortThreshold))
                               / (100 - self.rsiShortThreshold))

        self.logAll()
        return position

    def logAll(self):
        if self.logFilePath is None:
            return

        with open(self.logFilePath, "ab") as logFile:
            lowerBands = self.bbc.getLowerBands()
            upperBands = self.bbc.getUpperBands()
            sma = self.bbc.getSma()
            rsi = self.rsiC.getRsi()
            dayPrices = self.prices[:, -1]

            if lowerBands is None or upperBands is None or sma is None or rsi is None or dayPrices is None:
               return

            log = np.hstack([
                lowerBands.reshape(-1, 1),
                upperBands.reshape(-1, 1),
                sma.reshape(-1, 1),
                rsi.reshape(-1, 1),
                dayPrices.reshape(-1, 1)
            ])

            np.save(logFile, log)