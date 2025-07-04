from typing import List

import keras.models
import numpy as np

prices = None
positions = np.zeros(50)
isInnit = True
model = keras.models.load_model("best_model_from_GreeksNNTuning.keras")
greeksManager = None

LAGS = [1, 2, 3, 4, 5]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global positions, prices, greeksManager

    prices = prcSoFar

    if greeksManager is None:
        greeksManager = createGreeksManager()

    positions = updatePositions(prcSoFar[:, -1], greeksManager)

    return positions

def updatePositions(newDayPrices, gm) -> np.ndarray:
    gm.updateGreeks(newDayPrices.reshape(-1, 1))
    greeksData = gm.getGreeks().T
    greeksData = np.concatenate([greeksData, newDayPrices.reshape(-1, 1)], axis=1)

    greeksData = greeksData.flatten().reshape(1, -1)

    predictedLogReturns = model.predict(greeksData)[0].reshape(1, -1)[0]

    tradable_indices = [2, 4, 6, 10, 12, 14, 16, 21, 22, 25, 29, 32]

    newPositions = np.zeros_like(predictedLogReturns)

    for i in tradable_indices:
        if predictedLogReturns[i] > 0:
            newPositions[i] = 33333
        elif predictedLogReturns[i] < 0:
            newPositions[i] = -33333

    return newPositions

def createGreeksManager():
    lagged_prices_greeks = [LaggedPrices(prices, lag) for lag in LAGS]
    vol_greeks = [Volatility(prices, window) for window in VOL_WINDOWS]
    momentum_greeks = [Momentum(prices, window) for window in MOMENTUM_WINDOWS]
    greeks = (
            lagged_prices_greeks +
            vol_greeks +
            momentum_greeks +
            [
                LogReturns(prices, 1),
            ])
    gm = GreeksManager(greeks)
    return gm

class Greek:
    def update(self, newDayPrices: np.ndarray):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeks(self):
        raise NotImplementedError("Must override run() in subclass")

class LogReturns(Greek):
    def __init__(self, pricesSoFar: np.ndarray, lookback=1):
        super().__init__()

        self.lookback = lookback
        # n day lookback --> need n + 1 days of data (current and n prev)
        pricesFillWindow = pricesSoFar.shape[1] > lookback

        self.prices = pricesSoFar[:, -(lookback + 1):] if pricesFillWindow else pricesSoFar
        self.logReturns = np.full(self.prices.shape[0], np.nan)

        if pricesFillWindow:
            self.setLogReturns()

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        if self.prices.shape[1] > self.lookback:
            self.prices = self.prices[:, -(self.lookback + 1):]
            self.setLogReturns()

    def setLogReturns(self):
        lookbackPrices = self.prices[:, 0]
        currPrices = self.prices[:, -1]

        divByZeroMask = (lookbackPrices > 0) & (currPrices > 0)

        self.logReturns = np.full(lookbackPrices.shape, np.nan)
        self.logReturns[divByZeroMask] = np.log(currPrices[divByZeroMask] / lookbackPrices[divByZeroMask])

    def getGreeks(self):
        return self.logReturns

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

class LaggedPrices(Greek):
    def __init__(self, pricesSoFar: np.ndarray, lag):
        super().__init__()

        self.lag = lag
        pricesFillWindow = pricesSoFar.shape[1] > lag

        self.prices = pricesSoFar[:, -(lag + 1):] if pricesFillWindow else pricesSoFar

        if pricesFillWindow:
            self.lagPrices = pricesSoFar[:, -(lag + 1)]
        else:
            self.lagPrices = np.full(self.prices.shape[0], np.nan)

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices])

        if self.prices.shape[1] > self.lag + 1:
            self.prices = self.prices[:, -(self.lag + 1):]

        if self.prices.shape[1] > self.lag:
            self.lagPrices = self.prices[:, 0]

    def getGreeks(self):
        return self.lagPrices

class GreeksManager(Greek):
    def __init__(self, greeks: List[Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def update(self, newDayPrices: np.ndarray):
        for greek in self.greeks:
            greek.update(newDayPrices)

    def getGreeksList(self) -> List[Greek]:
        return self.greeks

    def getGreeks(self):
        greeksList = self.greeks
        greeksData = []

        for i, greek in enumerate(greeksList):
            greeksData.append(greek.getGreeks())

        return np.array(greeksData)