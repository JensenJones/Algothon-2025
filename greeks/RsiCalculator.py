import numpy as np


class RsiCalculator:
    def __init__(self, pricesSoFar, windowSize):
        pricesFillWindow: bool = pricesSoFar.shape[1] >= windowSize

        self.prices = pricesSoFar[:, -windowSize:] if pricesFillWindow else pricesSoFar
        self.windowSize = windowSize
        self.avgGain = None
        self.avgLoss = None
        self.rsi = None

        if pricesFillWindow:
            self.initialiseAverages()

    def updateWithNewDay(self, newDayPrices):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])

        if self.prices.shape[1] < self.windowSize:
            return

        self.prices = self.prices[:, -self.windowSize:]

        self.calculateRsi(newDayPrices)

    def initialiseAverages(self):
        delta = np.diff(self.prices[:, -self.windowSize:], axis=1)
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