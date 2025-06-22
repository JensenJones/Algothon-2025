import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

#TODO calculate

instrumentCount = 50
currentPos = np.zeros(instrumentCount)
bbc = None
trader = None
WINDOW_SIZE = 20

def getMyPosition(prcSoFar):
    global currentPos, instrumentCount, bbc, trader, WINDOW_SIZE

    day = prcSoFar.shape[1]

    if day < WINDOW_SIZE:
        return currentPos

    if trader is None:
        filePath = f"./strategies/bollBandsLog_WindowSize={WINDOW_SIZE}.npy"
        bbc = BollingerBandsCalculator(prcSoFar, WINDOW_SIZE, filePath)
        trader = BollingerBandsTrader(bbc, 500)

    currentPos = trader.updatePosition(prcSoFar[:, -1], currentPos)

    return currentPos


class BollingerBandsCalculator:
    """
    Middle band is the standard moving average (SMA) of the instrument price over D days.
    Upper band is the SMA + standard deviation of closing price over D days
    Lower Band is the SMA - standard deviation of closing price over D days
    """

    def __init__(self, prices: np.ndarray, windowSize: int, logFilePath: str):
        super().__init__()

        self.prices = prices
        self.WINDOW_SIZE = windowSize
        self.windowSum = None
        self.windowElements = None
        self.sma = None
        self.queueIndex = 0
        self.LOG_FILE_PATH = logFilePath

        self.calcSma()
        self.bollingerBands = np.zeros((50, 2)) # [:, 1] = upperBand, [:, 0] = lowerBand
        self.calcBollingerBands()

        open(logFilePath, "wb").close()
        self.logBollingerBandsAndPrice()


    def updateWithNewDay(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.updateSma(newDayPrices)
        self.calcBollingerBands()

        self.logBollingerBandsAndPrice()

    def calcSma(self):
        self.windowElements = self.prices[:, -self.WINDOW_SIZE:].copy()
        self.windowSum = np.sum(self.windowElements, axis = 1)
        self.sma = self.windowSum / self.WINDOW_SIZE

    def updateSma(self, newDayPrices: np.ndarray):
        self.windowSum -= self.windowElements[:, self.queueIndex]
        self.windowElements[:, self.queueIndex] = newDayPrices
        self.windowSum += newDayPrices

        self.queueIndex = (self.queueIndex + 1) % self.WINDOW_SIZE

        self.sma = self.windowSum / self.WINDOW_SIZE

    def calcBollingerBands(self):
        stDev = np.std(self.windowElements, axis = 1)
        self.bollingerBands[:, 1] = self.sma + 2 * stDev
        self.bollingerBands[:, 0] = self.sma - 2 * stDev

    def getUpperBands(self):
        return self.bollingerBands[:, 1]

    def getLowerBands(self):
        return self.bollingerBands[:, 0]

    def logBollingerBandsAndPrice(self):
        with open(self.LOG_FILE_PATH, "ab") as logFile:
            # write to file each instruments upper, lower, middle bands as well as price
            log = np.hstack([
                self.bollingerBands,                # shape: (50, 2)
                self.sma.reshape(-1, 1),            # shape: (50, 1
                self.prices[:, -1].reshape(-1, 1)   # shape: (50, 1)
            ])

            np.save(logFile, log)


class BollingerBandsTrader:
    def __init__(self, bollingerBandsCalculator, quantity: int):
        super().__init__()

        self.bbc = bollingerBandsCalculator
        self.quantity = quantity

    def updatePosition(self, newDayPrices: np.ndarray, position: np.ndarray):
        self.bbc.updateWithNewDay(newDayPrices)

        upperBands = self.bbc.getUpperBands()
        lowerBands = self.bbc.getLowerBands()


        position[newDayPrices < lowerBands] += self.quantity
        position[newDayPrices > upperBands] -= self.quantity

        return position