import numpy as np

from greeks.GreekBaseClass import Greek


class RollingMeans(Greek):
    def __init__(self, pricesSoFar: np.ndarray, windowSize = 14):
        super().__init__()

        self.windowSize = windowSize
        pricesFillWindow = pricesSoFar.shape[1] >= windowSize
        self.prices = pricesSoFar[:, -windowSize:] if pricesFillWindow else pricesSoFar

        if pricesFillWindow:
            self.rollingMean = np.mean(self.prices, axis = 1)
        else:
            self.rollingMean = np.full(self.prices.shape[0], np.nan)

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))

        if self.prices.shape[1] > self.windowSize:
            self.prices = self.prices[:, -self.windowSize:]

        self.rollingMean = np.mean(self.prices, axis = 1)

    def getGreeks(self):
        return self.rollingMean