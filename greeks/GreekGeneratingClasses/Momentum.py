import numpy as np
from greeks.GreekGeneratingClasses.GreekBaseClass import Greek

class Momentum(Greek):
    def __init__(self, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__()

        pricesFillWindow = pricesSoFar.shape[1] >= windowSize
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(windowSize + 1):] if pricesFillWindow else pricesSoFar
        self.momentum = np.full(pricesSoFar.shape[0], np.nan)

        if pricesFillWindow:
            self.setMomentum()

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))

        if self.pricesSoFar.shape[1] >= self.windowSize:
            self.pricesSoFar = self.pricesSoFar[:, -(self.windowSize + 1):]
            self.setMomentum()

    def setMomentum(self):
        log_returns = np.log(self.pricesSoFar[:, 1:] / self.pricesSoFar[:, :-1])
        self.momentum = np.nansum(log_returns[:, -self.windowSize:], axis=1)

    def getGreeks(self):
        return self.momentum
