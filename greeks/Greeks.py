import numpy as np


class Greeks:
    HIST_VOL_WINDOW_SIZE = 80
    ROLLING_MEAN_WINDOW_SIZE = 30

    def calcLogReturns(self):
        self.logReturns = np.diff(np.log(self.prices), axis=1)

    def calcHistVols(self):
        rolling_window = self.logReturns[:, -self.HIST_VOL_WINDOW_SIZE:]
        self.histVols = np.std(rolling_window, axis = 1, keepdims = True)

    def calcRollingMeans(self):
        pass

    def calcSpreads(self):
        pass

    def __init__(self, prices: np.ndarray):
        super().__init__()

        self.prices = prices
        self.logReturns = None
        self.histVols = None
        self.rollingMean = None
        self.spreads = None

        self.prices = prices
        self.calcLogReturns()
        self.calcHistVols()
        self.calcRollingMeans()
        self.calcSpreads()

    def updateWithNewDay(self, newDayPrices: np.ndarray) -> None:
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.updateLogReturns()
        self.updateHistVols()
        self.updateRollingMeans()
        self.updateSpreads()

    def updateLogReturns(self):
        prevDayPrices = self.prices[:, -2]
        newDayPrices = self.prices[:, -1]
        newLogReturns = (np.log(newDayPrices) - np.log(prevDayPrices)).reshape(-1, 1)
        self.logReturns = np.hstack([self.logReturns, newLogReturns])

    def updateHistVols(self):
        recentReturns = self.logReturns[:, -self.HIST_VOL_WINDOW_SIZE]
        new_vol = np.std(recentReturns, axis = 1, keepdims = True)
        self.histVols = np.hstack([self.histVols, new_vol])

    def updateRollingMeans(self):
        pass

    def updateSpreads(self):
        pass