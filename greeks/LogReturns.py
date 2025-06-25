import numpy as np

from greeks.GreekBaseClass import Greek


class LogReturns(Greek):
    def __init__(self, pricesSoFar: np.ndarray):
        super().__init__()

        self.prices = pricesSoFar[:, -2:] if pricesSoFar.shape[1] > 2 else pricesSoFar.copy()
        self.logReturns = np.full(self.prices.shape[0], np.nan)

        self.setDayIncrease()

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        if self.prices.shape[1] > 2:
            self.prices = self.prices[:, -2:]

        self.setDayIncrease()

    def setDayIncrease(self):
        if self.prices.shape[1] < 2:
            self.logReturns = np.zeros(self.prices.shape[0])
        else:
            prevPrices = self.prices[:, -2]
            currPrices = self.prices[:, -1]

            divByZeroMask = (prevPrices > 0) & (currPrices > 0)

            self.logReturns = np.full(prevPrices.shape, np.nan)
            self.logReturns[divByZeroMask] = np.log(currPrices[divByZeroMask] /
                                                    prevPrices[divByZeroMask])

    def getGreeks(self):
        return self.logReturns