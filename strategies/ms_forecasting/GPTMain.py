# Fixed Greek forecasting module with corrected log return indexing
import time

import numpy as np
import pandas as pd
import warnings
from skforecast.exceptions import MissingValuesWarning
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance


class Greek:
    def __init__(self, historyWindowSize):
        self.historyWindowSize = historyWindowSize + 1  # +1 to allow slicing for returns/history

    def update(self, newDayPrices: np.ndarray):
        raise NotImplementedError

    def getGreeks(self):
        raise NotImplementedError

    def getGreeksHistory(self):
        raise NotImplementedError


class Momentum(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(self.historyWindowSize + windowSize):]
        self.history = []
        for start in range(self.historyWindowSize):
            end = start + windowSize
            m = np.log(self.pricesSoFar[:, end] / self.pricesSoFar[:, start])
            self.history.append(m)
        self.history = np.stack(self.history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices[:, None]))
        self.pricesSoFar = self.pricesSoFar[:, 1:]
        momentum = np.log(
            self.pricesSoFar[:, -1] / self.pricesSoFar[:, -(self.windowSize + 1)]
        )
        self.history = np.hstack((self.history[:, 1:], momentum[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class Volatility(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize=5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(self.historyWindowSize + windowSize):]
        self.history = []
        for i in range(self.historyWindowSize):
            window = self.pricesSoFar[:, i : i + windowSize + 1]
            logr = np.log(window[:, 1:] / window[:, :-1])
            vol = np.std(logr, axis=1, ddof=1)
            self.history.append(vol)
        self.history = np.stack(self.history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices[:, None]))
        self.pricesSoFar = self.pricesSoFar[:, 1:]
        window = self.pricesSoFar[:, -self.windowSize - 1 :]
        logr = np.log(window[:, 1:] / window[:, :-1])
        vol = np.std(logr, axis=1, ddof=1)
        self.history = np.hstack((self.history[:, 1:], vol[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class LaggedPrices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, lag: int):
        super().__init__(historyWindowSize)
        self.lag = lag
        self.prices = pricesSoFar[:, -(self.historyWindowSize + lag):]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices[:, None]))
        self.prices = self.prices[:, 1:]

    def getGreeks(self):
        return self.prices[:, -(self.lag + 1)]

    def getGreeksHistory(self):
        return self.prices[:, : self.historyWindowSize]


class Prices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray):
        super().__init__(historyWindowSize)
        self.prices = pricesSoFar[:, -self.historyWindowSize :]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices[:, None]))
        self.prices = self.prices[:, 1:]

    def getGreeks(self):
        return self.prices[:, -1]

    def getGreeksHistory(self):
        return self.prices


class GreeksManager:
    def __init__(self, greeks: dict[str, Greek]):
        self.greeks = greeks

    def updateGreeks(self, newDayPrices: np.ndarray):
        for g in self.greeks.values():
            g.update(newDayPrices)

    def getGreeksDict(self, index: pd.Index) -> dict[str, pd.DataFrame]:
        names, objs = zip(*self.greeks.items())
        arr = np.concatenate([g.getGreeks()[:, None] for g in objs], axis=1)
        return {
            f"inst_{i}": pd.DataFrame(arr[i : i + 1, :], index=index, columns=names)
            for i in range(arr.shape[0])
        }

    def getGreeksHistoryDict(self, index: pd.Index) -> dict[str, pd.DataFrame]:
        names, objs = zip(*self.greeks.items())
        hist = np.concatenate(
            [np.swapaxes(g.getGreeksHistory(), 0, 1)[:-1, :, None] for g in objs],
            axis=2,
        )
        return {
            f"inst_{i}": pd.DataFrame(hist[:, i, :], index=index, columns=names)
            for i in range(hist.shape[1])
        }


# Forecaster setup
logReturnsForecaster = ForecasterRecursiveMultiSeries(
    regressor=HistGradientBoostingRegressor(
        random_state=8523, learning_rate=0.05, max_iter=400, min_samples_leaf=3
    ),
    transformer_series=None,
    transformer_exog=StandardScaler(),
    lags=50,
    window_features=RollingFeatures(stats=['min', 'max'], window_sizes=50),
)

# Greek windows
PRICE_LAGS = [lag for lag in range(1, 8)]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14, 20]

# Global state
nInst = 50
positions = np.zeros(nInst)
prices: np.ndarray = None
greeksManager: GreeksManager = None
firstInit = True
logReturns: pd.DataFrame = None
currentDay: int = None

# Parameters
TRAINING_MOD = 20
SIMPLE_THRESHOLD = 0.00
TRAINING_WINDOW_SIZE = 500
predictedLogReturnsHistory = []

def getMyPosition(prcSoFar: np.ndarray):
    global prices, greeksManager, firstInit, currentDay
    prices = prcSoFar
    newDayPrices = prcSoFar[:, -1]
    day = prices.shape[1]
    currentDay = day - 1

    if day < TRAINING_WINDOW_SIZE + max(PRICE_LAGS + VOL_WINDOWS + MOMENTUM_WINDOWS):
        return positions

    if firstInit:
        greeksManager = createGreeksManager(prices)
        updateLogReturns(prices)
        fitForecaster()
        firstInit = False
    else:
        greeksManager.updateGreeks(newDayPrices)
        updateLogReturns(prices)

    if day % TRAINING_MOD == 0:
        fitForecaster()

    predictedLogReturns = getPredictedLogReturns(1)
    updatePositions(predictedLogReturns)

    if day == 999:
        toLog = np.vstack(predictedLogReturnsHistory)
        np.save(
            "./strategies/ms_forecasting/predicted_log_returns_days_750-1000.npy", toLog
        )

    return positions


def fitForecaster():
    global logReturns
    exogDict = greeksManager.getGreeksHistoryDict(logReturns.index)
    logReturnsForecaster.fit(series=logReturns, exog=exogDict)


def updateLogReturns(prices=np.asarray([[0]])):
    global logReturns, currentDay
    currentDay = prices.shape[1] - 1
    # include one extra day for returns
    window = prices[:, -(TRAINING_WINDOW_SIZE + 1):]
    logr_np = np.log(window[:, 1:] / window[:, :-1])
    # returns correspond to days currentDay-TRAINING_WINDOW_SIZE+1 through currentDay
    start_day = currentDay - TRAINING_WINDOW_SIZE + 1
    idx = pd.RangeIndex(start=start_day, stop=currentDay + 1)
    logReturns = pd.DataFrame(
        logr_np.T,
        index=idx,
        columns=[f"inst_{i}" for i in range(logr_np.shape[0])]
    )
    return logReturns


def updatePositions(predictedLogReturns):
    global positions
    for i, val in enumerate(predictedLogReturns):
        if np.isnan(val):
            continue
        positions[i] = 50000 * (val / max(abs(val), 1e-9))


def getPredictedLogReturns(steps) -> np.ndarray:
    global currentDay
    future_idx = pd.RangeIndex(start=currentDay, stop=currentDay + steps)
    exogDict = greeksManager.getGreeksDict(future_idx)
    expected = np.log(prices[:, -1] / prices[:, -2])
    actual = logReturns.iloc[-1].to_numpy()
    np.testing.assert_array_almost_equal(expected, actual, decimal=12)
    pred = logReturnsForecaster.predict(
        steps=steps,
        last_window=logReturns.tail(max(logReturnsForecaster.lags)),
        exog=exogDict,
        levels=list(logReturns.columns),
    )
    pvals = pred["pred"].values
    predictedLogReturnsHistory.append(pvals)
    return pvals


def createGreeksManager(prices=np.asarray([[0]]), T=TRAINING_WINDOW_SIZE):
    lagged = {f"greek_lag_{l}": LaggedPrices(T, prices, l) for l in PRICE_LAGS}
    volat = {f"greek_volatility_{w}": Volatility(T, prices, w) for w in VOL_WINDOWS}
    moment = {f"greek_momentum_{w}": Momentum(T, prices, w) for w in MOMENTUM_WINDOWS}
    base = {"greek_price": Prices(T, prices)}
    return GreeksManager(lagged | volat | moment | base)


def getGreeksManagerForTesting():
    return greeksManager
