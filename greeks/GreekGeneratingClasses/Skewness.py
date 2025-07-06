from greeks.GreekGeneratingClasses.GreekBaseClass import Greek
import numpy as np

class Skewness(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(historyWindowSize + windowSize):]
        self.history = []

        for i in range(self.historyWindowSize):
            window = self.pricesSoFar[:, i:i+windowSize+1]
            log_returns = np.log(window[:, 1:] / window[:, :-1])
            skew = ((log_returns - log_returns.mean(axis=1, keepdims=True))**3).mean(axis=1)
            skew /= (np.std(log_returns, axis=1, ddof=1)**3 + 1e-9)
            self.history.append(skew)

        self.history = np.stack(self.history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        window = self.pricesSoFar[:, -self.windowSize-1:]
        log_returns = np.log(window[:, 1:] / window[:, :-1])
        skew = ((log_returns - log_returns.mean(axis=1, keepdims=True))**3).mean(axis=1)
        skew /= (np.std(log_returns, axis=1, ddof=1)**3 + 1e-9)
        self.history = np.hstack((self.history[:, 1:], skew[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history
