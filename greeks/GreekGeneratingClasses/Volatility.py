import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class Volatility(Greek):
    def __init__(self, pricesSoFar, windowSize = 5):
        super().__init__()

        pricesFillWindow = pricesSoFar.shape[1] >= windowSize
        self.pricesSoFar = pricesSoFar[:, -(windowSize + 1):] if pricesFillWindow else pricesSoFar
        self.windowSize = windowSize
        self.vols = np.full(pricesSoFar.shape[0], np.nan)

        if pricesFillWindow:
            self.setVols()

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))

        if self.pricesSoFar.shape[1] >= self.windowSize:
            self.pricesSoFar = self.pricesSoFar[:, -self.windowSize:]
            self.setVols()

    def setVols(self):
        log_returns = np.log(self.pricesSoFar[:, 1:] / self.pricesSoFar[:, :-1])
        self.vols = np.std(log_returns, axis=1, ddof=1)


    def getGreeks(self):
        return self.vols
