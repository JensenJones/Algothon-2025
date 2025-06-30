import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class LogReturns(Greek):
    def __init__(self, pricesSoFar: np.ndarray, lookback = 1):
        super().__init__()

        self.lookback = lookback
        # n day lookback --> need n + 1 days of data (current and n prev)
        pricesFillWindow = pricesSoFar.shape[1] > lookback

        self.prices = pricesSoFar[:, -(lookback + 1):] if pricesFillWindow else pricesSoFar
        self.logReturns = np.full(self.prices.shape[0], np.nan)

        if pricesFillWindow:
            self.setLogReturns()

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        if self.prices.shape[1] > self.lookback:
            self.prices = self.prices[:, -(self.lookback + 1):]
            self.setLogReturns()

    def setLogReturns(self):
        lookbackPrices = self.prices[:, 0]
        currPrices = self.prices[:, -1]

        divByZeroMask = (lookbackPrices > 0) & (currPrices > 0)

        self.logReturns = np.full(lookbackPrices.shape, np.nan)
        self.logReturns[divByZeroMask] = np.log(currPrices[divByZeroMask] / lookbackPrices[divByZeroMask])

    def getGreeks(self):
        return self.logReturns