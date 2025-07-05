import numpy as np
from greeks.GreekGeneratingClasses.GreekBaseClass import Greek

class Momentum(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(historyWindowSize + windowSize):]
        self.momentum = np.full(pricesSoFar.shape[0], np.nan)

        self.history = []
        for i in range(self.historyWindowSize):
            start = i
            end = i + windowSize + 1
            window = self.pricesSoFar[:, start:end]

            log_returns = np.log(window[:, 1:] / window[:, :-1])
            momentum = np.nansum(log_returns, axis=1)
            self.history.append(momentum)

        self.history = np.stack(self.history, axis=1)  # shape: (nInst, historyWindowSize)
        self.momentum = self.history[:, -1]

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # Calculate current momentum
        window = self.pricesSoFar[:, -self.windowSize-1:]
        log_returns = np.log(window[:, 1:] / window[:, :-1])
        momentum = np.nansum(log_returns, axis=1)
        self.momentum = momentum

        self.history = np.hstack((self.history[:, 1:], momentum[:, np.newaxis]))

    def getGreeks(self):
        return self.momentum

    def getGreeksHistory(self):
        return self.history