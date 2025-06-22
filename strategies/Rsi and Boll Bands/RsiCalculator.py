import numpy as np


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