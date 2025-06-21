import numpy as np


# I don't know why I made this, I don't have any use for it yet, it's a bit messy
class Greeks:
    def __init__(self, prices: np.ndarray, histVolWindowSize, rollingMeanWindowSize):
        super().__init__()

        self.prices = prices
        self.HIST_VOL_WINDOW_SIZE = histVolWindowSize
        self.ROLLING_MEAN_WINDOW_SIZE = rollingMeanWindowSize

        self.logReturns = []
        self.histVols = []
        self.rollingMean = []
        self.spreads = []

        self.calcLogReturns()
        self.calcHistVols()
        self.calcRollingMeans()
        self.calcSpreads()

    def updateWithNewDay(self, newDayPrices: np.ndarray) -> None:
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.updateLogReturns()
        self.calcHistVols()
        self.calcRollingMeans()
        self.calcSpreads()

    def calcLogReturns(self):
        self.logReturns.append(np.diff(np.log(self.prices), axis=1))

    def updateLogReturns(self):
        prevDayPrices = self.prices[:, -2]
        newDayPrices = self.prices[:, -1]
        newLogReturns = (np.log(newDayPrices) - np.log(prevDayPrices)).reshape(-1, 1)
        self.logReturns.append(newLogReturns)

    def calcHistVols(self):
        rollingWindow = self.logReturns[:, -self.HIST_VOL_WINDOW_SIZE:]
        self.histVols.append(np.std(rollingWindow, axis = 1, keepdims = True))

    def calcRollingMeans(self):
        self.rollingMean.append(np.mean(self.prices[:, -self.ROLLING_MEAN_WINDOW_SIZE:], axis = 1, keepdims = True))

    def calcSpreads(self):
        self.spreads.append(self.prices[:, np.newaxis, :] - self.prices[np.newaxis, :, :])

