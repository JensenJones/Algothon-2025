# Load
from typing import List

import joblib
import numpy as np
import pandas as pd
from skforecast.model_selection import TimeSeriesFold
import warnings
from skforecast.exceptions import MissingValuesWarning

warnings.simplefilter("ignore", category=MissingValuesWarning)

modelFilePath = "./saved models/forecaster_model_date-2025-07-04_time-13-24-17.pkl"
model_package = joblib.load(modelFilePath)

# For Greeks
PRICE_LAGS = [1, 2, 3, 4, 5]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]

# Unpack Model
best_params = model_package["best_params"]
best_lags = model_package["best_lags"]
# transformer_exog = model_package["transformer_exog"]
# window_features = model_package["window_features"]
forecaster = model_package["forecaster"]
feature_names = model_package["feature_names"]


forecaster.dropna_from_series = False

nInst = 50
positions = np.zeros(nInst)
prices = None
greeksManager = None
firstInit = True

TRAINING_MOD = 5
SIMPLE_THRESHOLD = 0.00
TRAINING_WINDOW_SIZE = 750

def getMyPosition(prcSoFar: np.ndarray): # TODO ---- This is the function that they call
    global prices, greeksManager, firstInit

    prices = prcSoFar
    newDayPrices = prcSoFar[:, -1] # shape (50, 1)
    day = prices.shape[1]

    if day < TRAINING_WINDOW_SIZE + max(PRICE_LAGS + VOL_WINDOWS + MOMENTUM_WINDOWS):
        return positions

    if firstInit:
        greeksManager = createGreeksManager()
        initialiseWithPrices()
        firstInit = False
    elif day % TRAINING_MOD == 0:
        greeksManager.updateGreeks(newDayPrices)

    fitForecaster()
    predictedLogReturns = getPredictedLogReturns()

    updatePositions(predictedLogReturns)

    return positions

def fitForecaster():
    logReturnsSoFar = np.log(prices[:, 1:] / prices[:, :-1])

    seriesDict = getSeriesDict(logReturnsSoFar)

    forecaster.fit(series=seriesDict, exog=greeksManager.getGreeksHistoryDict())

def updatePositions(predictedLogReturns):
    global positions

    for inst, predictedLogReturn in enumerate(predictedLogReturns):
        if np.isnan(predictedLogReturn):
            continue

        if predictedLogReturn > SIMPLE_THRESHOLD:
            strength = predictedLogReturn - SIMPLE_THRESHOLD
            positions[inst] = 50000 * strength
        elif predictedLogReturn < -SIMPLE_THRESHOLD:
            strength = predictedLogReturn + SIMPLE_THRESHOLD
            positions[inst] = 50000 * strength
        else:
            pass

def getPredictedLogReturns() -> pd.DataFrame:
    return forecaster.predict(
        steps   = 1,
        exog    = greeksManager.getGreeksHistoryDict(),
        levels  = None,
    )["pred"].values

def getSeriesDict(logReturnsSoFar):
    T = logReturnsSoFar.shape[1]
    index = pd.date_range(start="2000-01-01", periods=T, freq="D")

    seriesDict = {
        f"inst_{i}": pd.Series(logReturnsSoFar[i], index=index, name=f"inst_{i}")
        for i in range(logReturnsSoFar.shape[0])
    }
    return seriesDict

def initialiseWithPrices():
    global greeksManager
    greeksManager = createGreeksManager()

def createGreeksManager():
    # Dictionary keys match those used in exog in training of the model so that the transformer can work correctly
    # TODO Havent got this working yet though

    laggedPricesPrefix  = "greek_lag_"
    momentumPrefix      = "greek_momentum_"
    volatilityPrefix    = "greek_volatility_"
    pricesString        = "price"

    laggedPricesDict = {
        f"{laggedPricesPrefix}{lag}": LaggedPrices(TRAINING_WINDOW_SIZE, prices, lag)
        for lag in PRICE_LAGS
    }
    volatilityDict = {
        f"{volatilityPrefix}{window}" : Volatility(TRAINING_WINDOW_SIZE, prices, window)
        for window in VOL_WINDOWS
    }
    momentumDict = {
        f"{momentumPrefix}{window}" : Momentum(TRAINING_WINDOW_SIZE, prices, window)
        for window in MOMENTUM_WINDOWS
    }

    greeksDict = (
            laggedPricesDict |
            volatilityDict   |
            momentumDict     |
            {
                pricesString : Prices(TRAINING_WINDOW_SIZE, prices)
            }
    )

    assert len(greeksDict) == len(feature_names)
    gm = GreeksManager(greeksDict)
    return gm

# TODO check that these greeks are producing the correct history because the history part was all chat gpt

class Greek:
    def __init__(self, historyWindowSize):
        self.historyWindowSize = historyWindowSize

    def update(self, newDayPrices: np.ndarray):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeks(self):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeksHistory(self):
        raise NotImplementedError("Must override run() in subclass")

class Momentum(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.historyWindowSize = historyWindowSize
        self.history = []

        # Limit pricesSoFar to just enough for full backfill
        self.pricesSoFar = pricesSoFar[:, -(windowSize + historyWindowSize):]
        self.momentum = np.full(pricesSoFar.shape[0], np.nan)

        # Backfill history if we have enough price data
        if self.pricesSoFar.shape[1] >= windowSize + 1:
            for i in range(self.pricesSoFar.shape[1] - windowSize):
                window = self.pricesSoFar[:, i:i + windowSize + 1]
                log_returns = np.log(window[:, 1:] / window[:, :-1])
                momentum = np.nansum(log_returns, axis=1)
                self.history.append(momentum)
            self.momentum = self.history[-1]

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))

        if self.pricesSoFar.shape[1] >= self.windowSize + 1:
            self.pricesSoFar = self.pricesSoFar[:, -(self.windowSize + self.historyWindowSize):]
            self.setMomentum()
            if len(self.history) >= self.historyWindowSize:
                self.history.pop(0)
            self.history.append(self.momentum.copy())

    def setMomentum(self):
        log_returns = np.log(self.pricesSoFar[:, -self.windowSize:] / self.pricesSoFar[:, -(self.windowSize + 1):-1])
        self.momentum = np.nansum(log_returns, axis=1)

    def getGreeks(self):
        return self.momentum

    def getGreeksHistory(self):
        return np.array(self.history).T if self.history else np.empty((self.pricesSoFar.shape[0], 0))

class Volatility(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize=5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.historyWindowSize = historyWindowSize
        self.history = []

        # Trim to the last (windowSize + historyWindowSize) days if possible
        self.pricesSoFar = pricesSoFar[:, -(windowSize + historyWindowSize):]
        self.vols = np.full(pricesSoFar.shape[0], np.nan)

        # Backfill history if enough data
        if self.pricesSoFar.shape[1] >= windowSize + 1:
            for i in range(self.pricesSoFar.shape[1] - windowSize):
                window = self.pricesSoFar[:, i:i + windowSize + 1]
                log_returns = np.log(window[:, 1:] / window[:, :-1])
                vol = np.std(log_returns, axis=1, ddof=1)
                self.history.append(vol)
            self.vols = self.history[-1]

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))

        if self.pricesSoFar.shape[1] >= self.windowSize + 1:
            # Keep just enough prices for windowed volatility calculation
            self.pricesSoFar = self.pricesSoFar[:, -(self.windowSize + self.historyWindowSize):]
            self.setVols()
            if len(self.history) >= self.historyWindowSize:
                self.history.pop(0)
            self.history.append(self.vols.copy())

    def setVols(self):
        log_returns = np.log(self.pricesSoFar[:, -self.windowSize:] / self.pricesSoFar[:, -(self.windowSize + 1):-1])
        self.vols = np.std(log_returns, axis=1, ddof=1)

    def getGreeks(self):
        return self.vols

    def getGreeksHistory(self):
        return np.array(self.history).T if self.history else np.empty((self.pricesSoFar.shape[0], 0))

class LaggedPrices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, lag: int):
        super().__init__(historyWindowSize)
        self.lag = lag
        self.historyWindowSize = historyWindowSize
        self.history = []

        # Only keep what's needed
        self.prices = pricesSoFar[:, -(lag + historyWindowSize):]
        self.lagPrices = np.full(self.prices.shape[0], np.nan)

        # Backfill lagged prices history if enough data
        if self.prices.shape[1] > lag:
            for i in range(self.prices.shape[1] - lag):
                lagged = self.prices[:, i]
                self.history.append(lagged)
            self.lagPrices = self.history[-1]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))

        # Trim prices to keep only necessary columns
        if self.prices.shape[1] > self.lag + self.historyWindowSize:
            self.prices = self.prices[:, - (self.lag + self.historyWindowSize):]

        if self.prices.shape[1] > self.lag:
            self.lagPrices = self.prices[:, - (self.lag + 1)]
            if len(self.history) >= self.historyWindowSize:
                self.history.pop(0)
            self.history.append(self.lagPrices.copy())

    def getGreeks(self):
        return self.lagPrices

    def getGreeksHistory(self):
        return np.array(self.history).T if self.history else np.empty((self.prices.shape[0], 0))

class Prices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray):
        super().__init__(historyWindowSize)
        self.historyWindowSize = historyWindowSize

        # Trim to at most historyWindowSize
        self.prices = pricesSoFar[:, -historyWindowSize:]
        self.history = [self.prices[:, i].copy() for i in range(self.prices.shape[1])]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        if self.prices.shape[1] > self.historyWindowSize:
            self.prices = self.prices[:, -self.historyWindowSize:]

        if len(self.history) >= self.historyWindowSize:
            self.history.pop(0)
        self.history.append(newDayPrices.copy())

    def getGreeks(self):
        return self.prices[:, -1]

    def getGreeksHistory(self):
        return np.array(self.history).T if self.history else np.empty((self.prices.shape[0], 0))

class GreeksManager:
    def __init__(self, greeks: dict[str, Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def updateGreeks(self, newDayPrices: np.ndarray):
        for greek in self.greeks.values():
            greek.update(newDayPrices)

    def getGreeksHistoryDict(self) -> dict[str, pd.DataFrame]:
        nInst, T = next(iter(self.greeks.values())).getGreeksHistory().shape
        index = pd.date_range(start="2000-01-01", periods=T, freq="D")

        historyDict = {f"inst_{i}": pd.DataFrame(index=index) for i in range(nInst)}

        # Add Greek columns
        for name, greek in self.greeks.items():
            greekHist = greek.getGreeksHistory()  # shape: (nInst, T)
            for i in range(nInst):
                historyDict[f"inst_{i}"][name] = greekHist[i]

        for i in range(nInst):
            df = historyDict[f"inst_{i}"]
            try:
                df = df[feature_names]
            except KeyError:
                print(f"\n[ERROR] Columns found: {df.columns}")
                print(f"[ERROR] Columns expected: {feature_names}\n")
                raise
            historyDict[f"inst_{i}"] = df

        return historyDict