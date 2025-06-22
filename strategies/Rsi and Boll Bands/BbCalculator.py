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
