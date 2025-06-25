import numpy as np

from greeks.GreekBaseClass import Greek


class LaggedPrices(Greek):
    def __init__(self, pricesSoFar: np.ndarray, lag):
        super().__init__()

        self.lag = lag
        pricesFillWindow = pricesSoFar.shape[1] > lag

        self.prices = pricesSoFar[:, -(lag + 1):] if pricesFillWindow else pricesSoFar

        if pricesFillWindow:
            self.lagPrices = pricesSoFar[:, -(lag + 1)]
        else:
            self.lagPrices = np.full(self.prices.shape[0], np.nan)


    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices])

        if self.prices.shape[1] > self.lag + 1:
            self.prices = self.prices[:, -(self.lag + 1):]

        if self.prices.shape[1] > self.lag:
            self.lagPrices = self.prices[:, 0]

    def getGreeks(self):
        return self.lagPrices