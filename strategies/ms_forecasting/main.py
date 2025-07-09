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
        self.historyWindowSize = historyWindowSize + 1 # + 1 because we need to trim off the most recent day on fit then
                                                       # still have window size left

    def update(self, newDayPrices: np.ndarray):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeks(self):
        raise NotImplementedError("Must override run() in subclass")

    def getGreeksHistory(self):
        raise NotImplementedError("Must override run() in subclass")

# PROBABLY CORRECT (95%)
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

# PROBABLY CORRECT (95%)
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

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # Calculate and store latest volatility
        window = self.pricesSoFar[:, -self.windowSize - 1:]
        window_logReturns = np.log(window[:, 1:] / window[:, :-1])

        assert window_logReturns.shape[1] == self.windowSize, f"BAD CALCULATION, window size = {window_logReturns.shape}"

        vol = np.std(window_logReturns, axis=1, ddof=1)

        self.history = np.hstack((self.history[:, 1:], vol[:, np.newaxis]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history

class LaggedPrices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, lag: int):
        super().__init__(historyWindowSize)
        self.lag = lag
        self.prices = pricesSoFar[:, (pricesSoFar.shape[1] - self.historyWindowSize - lag):]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack((self.prices, newDayPrices.reshape(-1, 1)))
        self.prices = self.prices[:, 1:]

    def getGreeks(self):
        return self.prices[:, -(self.lag + 1)]

    def getGreeksHistory(self):
        return self.prices[:, :self.historyWindowSize]

class Prices(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray):
        super().__init__(historyWindowSize)
        self.prices = pricesSoFar[:, -self.historyWindowSize:]

    def update(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.prices = self.prices[:, 1:]

    def getGreeks(self):
        return self.prices[:, -1]

    def getGreeksHistory(self):
        return self.prices

class GreeksManager:
    def __init__(self, greeks: dict[str, Greek]):
        self.greeks = greeks

    def updateGreeks(self, newDayPrices: np.ndarray):
        for greek in self.greeks.values():
            greek.update(newDayPrices)

    def getGreeksList(self):
        return self.greeks.values()

    def getGreeksDict(self, index: pd.Index) -> dict[str, pd.DataFrame]:
        feature_names, greek_objs = zip(*self.greeks.items())

        greeks_list = [
            greek.getGreeks().reshape(-1, 1)
            for greek in greek_objs
        ]

        greeks_array = np.concatenate(greeks_list, axis=1)

        greeks_dict = {}
        for inst_idx in range(greeks_array.shape[0]):
            greeks_dict[f"inst_{inst_idx}"] = pd.DataFrame(
                greeks_array[inst_idx:inst_idx+1, :],
                index=index,
                columns=feature_names
            )

        return greeks_dict

    def getGreeksHistoryDict(self, index: pd.Index) -> dict[str, pd.DataFrame]:
        feature_names, greek_objs = zip(*self.greeks.items())

        greek_history_list = [
            np.swapaxes(greek.getGreeksHistory(), 0, 1)[:-1, :, np.newaxis]
            for greek in greek_objs
        ]

        greek_history_array = np.concatenate(greek_history_list, axis=-1)

        greeks_history_dict = {}
        n_instruments = greek_history_array.shape[1]
        for inst_idx in range(n_instruments):
            greeks_history_dict[f"inst_{inst_idx}"] = pd.DataFrame(
                greek_history_array[:, inst_idx, :],
                index=index,
                columns=feature_names
            )

        return greeks_history_dict


# TODO assure that the correct greek names line up with the correct greeks data in dict and histDict

logReturnsForecaster = ForecasterRecursiveMultiSeries(
    regressor           = HistGradientBoostingRegressor(random_state=8523,
                                                        learning_rate=0.05),
    transformer_series  = None,
    transformer_exog    = StandardScaler(),
    lags                = 20,
    window_features     = RollingFeatures(
                                stats           = ['min', 'max'],
                                window_sizes    = 10,
                            ),
)

PRICE_LAGS = [lag for lag in range(1, 8)]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]

nInst = 50
positions = np.zeros(nInst)
prices: np.ndarray = None
greeksManager: GreeksManager = None
firstInit = True
logReturns: pd.DataFrame = None
currentDay: int = None

TRAINING_MOD = 20
SIMPLE_THRESHOLD = 0.00
TRAINING_WINDOW_SIZE = 500

predictedLogReturnsHistory = []

def getMyPosition(prcSoFar: np.ndarray): # This is the function that they call
    global prices, greeksManager, firstInit, currentDay

    prices = prcSoFar
    newDayPrices = prcSoFar[:, -1] # shape (50, 1)
    day = prices.shape[1]

    daysCount = prices.shape[1]
    currentDay = daysCount - 1

    if day < TRAINING_WINDOW_SIZE + max(PRICE_LAGS + VOL_WINDOWS + MOMENTUM_WINDOWS):
        print(f"Shouldn't be hitting this, prcSoFar.shape = {prcSoFar.shape}")
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

    # predictedLogReturns_10steps = getPredictedLogReturns(10)
    # predictedLogReturns_10steps = predictedLogReturns_10steps.reshape(10, 50)
    # predictedLogReturns = predictedLogReturns_10steps.sum(axis=1)

    predictedLogReturns = getPredictedLogReturns(1)
    updatePositions(predictedLogReturns)

    if day == 999:
        toLog = np.vstack(predictedLogReturnsHistory)
        np.save("./strategies/ms_forecasting/predicted_log_returns_days_750-1000.npy", toLog)

    return positions

def fitForecaster():
    global logReturns

    exogDict = greeksManager.getGreeksHistoryDict(logReturns.index) # current day is already trimmed from the exogs

    logReturnsForecaster.fit(
        series=logReturns,
        exog=exogDict,
    )

def updateLogReturns(prices = prices):
    global logReturns

    currentDay = prices.shape[1] - 1
    pricesInWindow = prices[:, -(TRAINING_WINDOW_SIZE + 1):]
    logReturnsSoFarNp = np.log(pricesInWindow[:, 1:] / pricesInWindow[:, :-1])

    index = pd.RangeIndex(start=currentDay - TRAINING_WINDOW_SIZE + 1, stop=currentDay + 1)

    logReturns = pd.DataFrame(logReturnsSoFarNp.T,
                              index = index,
                              columns = [f"inst_{i}" for i in range(logReturnsSoFarNp.shape[0])])

    return logReturns # Added for the testing

def updatePositions(predictedLogReturns):
    global positions

    for inst, predictedLogReturn in enumerate(predictedLogReturns):
        if np.isnan(predictedLogReturn):
            continue

        if predictedLogReturn > SIMPLE_THRESHOLD:
            strength = predictedLogReturn / max(SIMPLE_THRESHOLD, 1e-9)
            positions[inst] = 50000 * strength
        elif predictedLogReturn < -SIMPLE_THRESHOLD:
            strength = predictedLogReturn / max(SIMPLE_THRESHOLD, 1e-9)
            positions[inst] = 50000 * strength
        else:
            pass

def getPredictedLogReturns(steps) -> np.ndarray:
    global currentDay

    futureIndex = pd.RangeIndex(start=currentDay + 1,
                               stop=currentDay + 1 + steps)
    exogDict = greeksManager.getGreeksDict(futureIndex)

    prediction = logReturnsForecaster.predict(
        steps       = steps,
        last_window = logReturns.tail(max(logReturnsForecaster.lags)),
        exog        = exogDict,
        levels      = list(logReturns.columns),
    )

    predictedLogReturns = prediction["pred"].values

    predictedLogReturnsHistory.append(predictedLogReturns)

    return predictedLogReturns

def createGreeksManager(prices = prices):
    laggedPricesPrefix  = "greek_lag_"
    volatilityPrefix    = "greek_volatility_"
    momentumPrefix      = "greek_momentum_"
    pricesString        = "greek_price"

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

    gm = GreeksManager(greeksDict)

    return gm

def getGreeksManagerForTesting():
    return greeksManager