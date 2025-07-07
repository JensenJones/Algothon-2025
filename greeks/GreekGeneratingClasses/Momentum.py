import numpy as np
from greeks.GreekGeneratingClasses.GreekBaseClass import Greek

class Momentum(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(historyWindowSize + windowSize + 1):]
        self.history = []

        for startDay in range(self.historyWindowSize):
            endDay = startDay + windowSize

            momentum = np.log(self.pricesSoFar[:, endDay] / self.pricesSoFar[:, startDay])

            self.history.append(momentum)

        self.history = np.stack(self.history, axis=1)  # shape: (nInst, historyWindowSize)

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # Calculate current momentum
        startDay = -(self.windowSize + 1)
        endDay = -1

        assert startDay - endDay == -self.windowSize, f"Update is wrong, start = {startDay}, end = {endDay}"

        momentum = np.log(self.pricesSoFar[:, endDay] / self.pricesSoFar[:, startDay])

        self.history = np.hstack((self.history[:, 1:], momentum[:, np.newaxis]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history
