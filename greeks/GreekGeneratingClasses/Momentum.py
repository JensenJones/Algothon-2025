import numpy as np
from greeks.GreekGeneratingClasses.GreekBaseClass import Greek

class Momentum(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(historyWindowSize + windowSize):]
        self.history = []

        for startDay in range(self.historyWindowSize):
            endDay = startDay + windowSize - 1

            momentum = np.log(self.pricesSoFar[:, startDay] / self.pricesSoFar[:, endDay])

            self.history.append(momentum)

        self.history = np.stack(self.history, axis=1)  # shape: (nInst, historyWindowSize)

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # Calculate current momentum
        startDay = -self.windowSize
        endDay = -1

        assert startDay - endDay == -self.windowSize + 1, f"Update is wrong, start = {startDay}, end = {endDay}"

        momentum = np.log(self.pricesSoFar[:, startDay] / self.pricesSoFar[:, endDay])

        self.history = np.hstack((self.history[:, 1:], momentum[:, np.newaxis]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history
