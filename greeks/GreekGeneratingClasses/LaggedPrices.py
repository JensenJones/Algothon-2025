import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class LaggedPrices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, lag: int):
        super().__init__(historyWindowSize)
        self.lag = lag
        self.prices = pricesSoFar[:, (pricesSoFar.shape[1] - self.historyWindowSize - lag):]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        self.prices = self.prices[:, 1:]

    def getGreeks(self):
        return self.prices[:, -(self.lag + 1)]

    def getGreeksHistory(self):
        return self.prices[:, :self.historyWindowSize]