import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class Prices(Greek):
    def __init__(self, pricesSoFar: np.ndarray):
        super().__init__()

        self.prices = pricesSoFar

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])

    def getGreeks(self):
        return self.prices