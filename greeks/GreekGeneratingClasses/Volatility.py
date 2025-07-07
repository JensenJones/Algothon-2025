import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class Volatility(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize=5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(self.historyWindowSize + windowSize):]
        self.history = []

        for i in range(self.historyWindowSize):
            start = i
            end = i + windowSize + 1
            window = self.pricesSoFar[:, start:end]

            window_logReturns = np.log(window[:, 1:] / window[:, :-1])

            assert window_logReturns.shape[1] == windowSize, f"BAD CALCULATION, window size = {window_logReturns.shape}, i = {i}"

            vol = np.std(window_logReturns, axis=1, ddof=1)

            self.history.append(vol)

        self.history = np.stack(self.history, axis=1)  # (nInst, historyWindowSize)
        self.vols = self.history[:, -1]

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # Calculate and store latest volatility
        window = self.pricesSoFar[:, -self.windowSize - 1:]
        window_logReturns = np.log(window[:, 1:] / window[:, :-1])

        assert window_logReturns.shape[1] == self.windowSize, f"BAD CALCULATION, window size = {window_logReturns.shape}"

        vol = np.std(window_logReturns, axis=1, ddof=1)

        self.history = np.hstack((self.history[:, 1:], vol[:, np.newaxis]))
        self.vols = vol

    def getGreeks(self):
        return self.vols

    def getGreeksHistory(self):
        return np.array(self.history)