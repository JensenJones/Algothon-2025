import numpy as np

from greeks.Greek import Greek


class DayIncreases(Greek):
    def __init__(self, pricesSoFar: np.ndarray):
        super().__init__()

        self.prices = pricesSoFar[:, -2:] if pricesSoFar.shape[1] > 2 else pricesSoFar.copy()
        self.dayIncrease = None

        self.setDayIncrease()

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        if self.prices.shape[1] > 2:
            self.prices = self.prices[:, -2:]

        self.setDayIncrease()

    def setDayIncrease(self):
        if self.prices.size < 2:
            self.dayIncrease = None
        else:
            self.dayIncrease = np.diff(self.prices[:, -2:-1], axis = 1)

    def getGreeks(self):
        return self.dayIncrease