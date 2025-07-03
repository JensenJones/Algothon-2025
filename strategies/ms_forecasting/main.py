# Load
from typing import List

import joblib
import numpy as np
import pandas as pd
from skforecast.model_selection import TimeSeriesFold

modelFilePath = "./saved models/forecaster_model_date-2025-07-03_time-16-49-04.pkl"
model_package = joblib.load(modelFilePath)

# For Greeks
LAGS = [1, 2, 3, 4, 5]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]

# Unpack Model
forecaster = model_package["forecaster"]
best_params = model_package["best_params"]
best_lags = model_package["best_lags"]
transformer_exog = model_package["transformer_exog"]

assert forecaster.is_fitted == True, "Forecaster has not been fit"

nInst = 50
positions = np.zeros(nInst)
prices = None
greeksManager = None
firstInit = True

SIMPLE_THRESHOLD = 0.001

def getMyPosition(prcSoFar: np.ndarray): # TODO ---- This is the function that they call
    global prices, greeksManager, firstInit

    prices = prcSoFar
    newDayPrices = prcSoFar[:, -1] # shape (50, 1)

    if firstInit:
        greeksManager = createGreeksManager()
        initialiseWithPrices()
        firstInit = False
    else:
        greeksManager.updateGreeks(newDayPrices)

    fitForecaster()

    predictedLogReturns = getPredictedLogReturns()

    updatePositions(predictedLogReturns)

    return positions

def fitForecaster():
    logReturnsSoFar = np.log(prices[:, 1:] / prices[:, :-1])

    seriesDict = getSeriesDict(logReturnsSoFar)

    exogDict = greeksToExogDict(greeksManager.getGreeksHistory())

    forecaster.fit(
        series  = seriesDict,
        exog    = exogDict,
    )

def updatePositions(predictedLogReturns):
    global positions

    for instrument, prediction in enumerate(predictedLogReturns):
        if np.isnan(prediction):
            continue

        if prediction > SIMPLE_THRESHOLD:
            strength = (prediction - SIMPLE_THRESHOLD) / SIMPLE_THRESHOLD
            positions[i] = 1000 * strength
        elif prediction < -SIMPLE_THRESHOLD:
            strength = (prediction + SIMPLE_THRESHOLD) / SIMPLE_THRESHOLD
            positions[i] = 1000 * strength
        else:
            positions[i] = 0

def getPredictedLogReturns() -> pd.DataFrame:
    return forecaster.predict(
        steps   = 1,
        exog    = greeksToExogDict(greeksManager.getGreeks()),
        levels  = None,
    )["pred"].values

def getSeriesDict(logReturnsSoFar):
    T = logReturnsSoFar.shape[1]
    index = pd.Index(range(T))  # simple integer index 0 to T-1
    seriesDict = {
        f"inst_{i}": pd.Series(logReturnsSoFar[i], index=index, name=f"inst_{i}")
        for i in range(logReturnsSoFar.shape[0])
    }
    return seriesDict

def greeksToExogDict(greeksHist: np.ndarray) -> dict:
    nInst, T, nFeat = greeksHist.shape
    feature_names = [f"feat_{i}" for i in range(nFeat)]

    return {
        f"inst_{i}": pd.DataFrame(greeksHist[i], columns=feature_names)
        for i in range(nInst)
    }

def initialiseWithPrices():
    global greeksManager
    greeksManager = createGreeksManager()

def createGreeksManager():
    lagged_prices_greeks = [LaggedPrices(prices, lag) for lag in LAGS]
    vol_greeks = [Volatility(prices, window) for window in VOL_WINDOWS]
    momentum_greeks = [Momentum(prices, window) for window in MOMENTUM_WINDOWS]
    greeks = (
            lagged_prices_greeks +
            vol_greeks +
            momentum_greeks +
            [
                LogReturns(prices, lookback=1),
                Prices(prices)
            ])
    gm = GreeksManager(greeks)
    return gm

class Greek:
    def update(self, newDayPrices: np.ndarray):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeks(self):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeksHistory(self):
        raise NotImplementedError("Must override run() in subclass")

class LogReturns(Greek):
    def __init__(self, pricesSoFar: np.ndarray, lookback=1):
        super().__init__()
        self.lookback = lookback
        self.history = []
        pricesFillWindow = pricesSoFar.shape[1] > lookback
        self.prices = pricesSoFar[:, -(lookback + 1):] if pricesFillWindow else pricesSoFar
        self.logReturns = np.full(self.prices.shape[0], np.nan)
        if pricesFillWindow:
            self.setLogReturns()
            self.history.append(self.logReturns.copy())

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        if self.prices.shape[1] > self.lookback:
            self.prices = self.prices[:, -(self.lookback + 1):]
            self.setLogReturns()
            self.history.append(self.logReturns.copy())

    def setLogReturns(self):
        lookbackPrices = self.prices[:, 0]
        currPrices = self.prices[:, -1]
        divByZeroMask = (lookbackPrices > 0) & (currPrices > 0)
        self.logReturns = np.full(lookbackPrices.shape, np.nan)
        self.logReturns[divByZeroMask] = np.log(currPrices[divByZeroMask] / lookbackPrices[divByZeroMask])

    def getGreeks(self):
        return self.logReturns

    def getGreeksHistory(self):
        return np.array(self.history).T

class Momentum(Greek):
    def __init__(self, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__()
        self.windowSize = windowSize
        self.history = []
        pricesFillWindow = pricesSoFar.shape[1] >= windowSize
        self.pricesSoFar = pricesSoFar[:, -(windowSize + 1):] if pricesFillWindow else pricesSoFar
        self.momentum = np.full(pricesSoFar.shape[0], np.nan)
        if pricesFillWindow:
            self.setMomentum()
            self.history.append(self.momentum.copy())

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        if self.pricesSoFar.shape[1] >= self.windowSize:
            self.pricesSoFar = self.pricesSoFar[:, -(self.windowSize + 1):]
            self.setMomentum()
            self.history.append(self.momentum.copy())

    def setMomentum(self):
        log_returns = np.log(self.pricesSoFar[:, 1:] / self.pricesSoFar[:, :-1])
        self.momentum = np.nansum(log_returns[:, -self.windowSize:], axis=1)

    def getGreeks(self):
        return self.momentum

    def getGreeksHistory(self):
        return np.array(self.history).T

class Volatility(Greek):
    def __init__(self, pricesSoFar, windowSize=5):
        super().__init__()
        self.windowSize = windowSize
        self.history = []
        pricesFillWindow = pricesSoFar.shape[1] >= windowSize
        self.pricesSoFar = pricesSoFar[:, -(windowSize + 1):] if pricesFillWindow else pricesSoFar
        self.vols = np.full(pricesSoFar.shape[0], np.nan)
        if pricesFillWindow:
            self.setVols()
            self.history.append(self.vols.copy())

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        if self.pricesSoFar.shape[1] >= self.windowSize:
            self.pricesSoFar = self.pricesSoFar[:, -self.windowSize:]
            self.setVols()
            self.history.append(self.vols.copy())

    def setVols(self):
        log_returns = np.log(self.pricesSoFar[:, 1:] / self.pricesSoFar[:, :-1])
        self.vols = np.std(log_returns, axis=1, ddof=1)

    def getGreeks(self):
        return self.vols

    def getGreeksHistory(self):
        return np.array(self.history).T

class LaggedPrices(Greek):
    def __init__(self, pricesSoFar: np.ndarray, lag):
        super().__init__()
        self.lag = lag
        self.history = []
        pricesFillWindow = pricesSoFar.shape[1] > lag
        self.prices = pricesSoFar[:, -(lag + 1):] if pricesFillWindow else pricesSoFar
        self.lagPrices = np.full(self.prices.shape[0], np.nan)
        if pricesFillWindow:
            self.lagPrices = pricesSoFar[:, -(lag + 1)]
            self.history.append(self.lagPrices.copy())

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        if self.prices.shape[1] > self.lag + 1:
            self.prices = self.prices[:, -(self.lag + 1):]
        if self.prices.shape[1] > self.lag:
            self.lagPrices = self.prices[:, 0]
            self.history.append(self.lagPrices.copy())

    def getGreeks(self):
        return self.lagPrices

    def getGreeksHistory(self):
        return np.array(self.history).T

class Prices(Greek):
    def __init__(self, pricesSoFar: np.ndarray):
        super().__init__()
        self.prices = pricesSoFar
        self.history = [pricesSoFar[:, -1].copy()]  # Latest day

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.history.append(newDayPrices.copy())

    def getGreeks(self):
        return self.prices[:, -1]

    def getGreeksHistory(self):
        return np.array(self.history).T

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

    def getGreeksHistory(self):
        all_histories = [g.getGreeksHistory() for g in self.greeks]  # each (nInst, T)
        return np.stack(all_histories, axis=-1)  # final: (nInst, T, nFeatures)