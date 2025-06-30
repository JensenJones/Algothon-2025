import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class BollingerBandsCalculator(Greek):
    def __init__(self, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__()

        pricesFillWindow = pricesSoFar.shape[1] >= windowSize

        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -windowSize:] if pricesFillWindow else pricesSoFar

        self.windowSize = windowSize
        self.sma = None
        self.queueIndex = 0
        self.bollingerBands = np.zeros((50, 2)) # [:, 1] = upperBand, [:, 0] = lowerBand

        self.calcSma()
        self.calcBollingerBands()

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack([self.pricesSoFar, newDayPrices.reshape(-1, 1)])

        if self.pricesSoFar.shape[1] < self.windowSize:
            return

        self.pricesSoFar = self.pricesSoFar[:, 1:]

        self.calcSma()
        self.calcBollingerBands()

    def calcSma(self):
        windowSum = np.sum(self.pricesSoFar, axis = 1)
        self.sma = windowSum / self.windowSize

    def calcBollingerBands(self):
        stDev = np.std(self.pricesSoFar, axis = 1)
        self.bollingerBands[:, 1] = self.sma + 2 * stDev
        self.bollingerBands[:, 0] = self.sma - 2 * stDev

    def getUpperBands(self):
        return self.bollingerBands[:, 1]

    def getLowerBands(self):
        return self.bollingerBands[:, 0]

    def getSma(self):
        return self.sma

    def getGreeks(self):
        pass