import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class Volatility(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize=5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(historyWindowSize + windowSize):]
        self.vols = np.full(pricesSoFar.shape[0], np.nan)
        self.history = []

        # Backfill exactly `historyWindowSize` values
        for i in range(self.historyWindowSize):
            start = i
            end = i + self.windowSize + 1
            window = self.pricesSoFar[:, start:end]

            if window.shape[1] <= 1:
                vol = np.full(window.shape[0], np.nan)
            else:
                log_returns = np.log(window[:, 1:] / window[:, :-1])
                vol = np.std(log_returns, axis=1, ddof=1)

            self.history.append(vol)

        self.history = np.stack(self.history, axis=1)  # (nInst, historyWindowSize)
        self.vols = self.history[:, -1]

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # Calculate and store latest volatility
        window = self.pricesSoFar[:, -self.windowSize - 1:]

        if window.shape[1] <= 1:
            vol = np.full(window.shape[0], np.nan)
        else:
            log_returns = np.log(window[:, 1:] / window[:, :-1])
            vol = np.std(log_returns, axis=1, ddof=1)

        self.history = np.hstack((self.history[:, 1:], vol[:, np.newaxis]))

    def getGreeks(self):
        return self.vols

    def getGreeksHistory(self):
        return np.array(self.history)
