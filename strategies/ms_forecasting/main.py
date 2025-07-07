import time

import numpy as np
import pandas as pd
import warnings
from skforecast.exceptions import MissingValuesWarning
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# TODO CHECK MOMENTUM
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

# PROBABLY CORRECT (90%)
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

# TODO CHECK SKEWNESS
class Skewness(Greek):
    def __init__(self, historyWindowSize, pricesSoFar: np.ndarray, windowSize: int):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        self.pricesSoFar = pricesSoFar[:, -(historyWindowSize + windowSize):]
        self.history = []

        for i in range(self.historyWindowSize):
            window = self.pricesSoFar[:, i:i+windowSize+1]
            log_returns = np.log(window[:, 1:] / window[:, :-1])
            skew = ((log_returns - log_returns.mean(axis=1, keepdims=True))**3).mean(axis=1)
            skew /= (np.std(log_returns, axis=1, ddof=1)**3 + 1e-9)
            self.history.append(skew)

        self.history = np.stack(self.history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        window = self.pricesSoFar[:, -self.windowSize-1:]
        log_returns = np.log(window[:, 1:] / window[:, :-1])
        skew = ((log_returns - log_returns.mean(axis=1, keepdims=True))**3).mean(axis=1)
        skew /= (np.std(log_returns, axis=1, ddof=1)**3 + 1e-9)
        self.history = np.hstack((self.history[:, 1:], skew[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history

class GreeksManager:
    def __init__(self, greeks: dict[str, Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def updateGreeks(self, newDayPrices: np.ndarray):
        for greek in self.greeks.values():
            greek.update(newDayPrices)

    def getGreeksList(self):
        return self.greeks.values()

    def getGreeksDict(self):
        greeksArray = [
            greek.getGreeks().reshape(-1, 1)
            for greek in self.greeks.values()
        ]

        greeksArrayNp = np.concatenate(greeksArray, axis=-1)  # shape (TRAINING_WINDOW_SIZE, nInst, num_greeks)
        featureNames = list(self.greeks.keys())

        exogDict = {
            f"inst_{i}": pd.DataFrame([greeksArrayNp[i, :]], columns=featureNames)
            for i in range(greeksArrayNp.shape[0])  # iterate over instruments
        }

        return exogDict

    def getGreeksHistoryDict(self) -> dict[str, pd.DataFrame]:
        greekHistoryArray = [
            np.swapaxes(greek.getGreeksHistory(), 0, 1)[:-1, :, np.newaxis]
            for greek in self.greeks.values()
        ]

        greekArrayNp = np.concatenate(greekHistoryArray, axis=-1)  # shape (days, nInst, num_greeks)
        featureNames = list(self.greeks.keys())

        exogDict = {
            f"inst_{i}": pd.DataFrame(greekArrayNp[:, i, :], columns=featureNames)
            for i in range(greekArrayNp.shape[1])
        }


        return exogDict

warnings.simplefilter("ignore", category=MissingValuesWarning)

logReturnsForecaster = ForecasterRecursiveMultiSeries(
    regressor           = HistGradientBoostingRegressor(random_state=8523, learning_rate=0.09),
    transformer_series  = None,
    transformer_exog    = StandardScaler(),
    lags                = 7,
    window_features     = RollingFeatures(
                                stats           = ['min', 'max'],
                                window_sizes    = 7,
                            )
)

PRICE_LAGS = [1, 2, 3, 4, 5, 6, 7]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]
SKEWNESS_WINDOWS = [5, 10]

logReturnsForecaster.dropna_from_series = True

nInst = 50
positions = np.zeros(nInst)
prices: np.ndarray = None
greeksManager: GreeksManager = None
firstInit = True
logReturns: pd.DataFrame = None

TRAINING_MOD = 7
SIMPLE_THRESHOLD = 0.00
TRAINING_WINDOW_SIZE = 600

predictedLogReturnsHistory = []

def getMyPosition(prcSoFar: np.ndarray): # This is the function that they call

    global prices, greeksManager, firstInit

    prices = prcSoFar
    newDayPrices = prcSoFar[:, -1] # shape (50, 1)
    day = prices.shape[1]

    if day < TRAINING_WINDOW_SIZE + max(PRICE_LAGS + VOL_WINDOWS + MOMENTUM_WINDOWS + SKEWNESS_WINDOWS):
        print(f"Shouldn't be hitting this, prcSoFar.shape = {prcSoFar.shape}")
        return positions

    if firstInit:
        greeksManager = createGreeksManager()
        updateLogReturns()
        fitForecaster()
        firstInit = False
    else:
        greeksManager.updateGreeks(newDayPrices)
        updateLogReturns()

    if day % TRAINING_MOD == 0:
        fitForecaster()
    # TODO ################################################################################# CHECK THE GREEKS

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

    # assert np.isclose(logReturns["inst_0"].iloc[-1], np.log(prices[0, -1] / prices[0, -2])), \
        # f"TRAINING IS NOT THE SAME, recent = {logReturns["inst_0"].iloc[-1]}\nwrong is {np.log(prices[0, -1] / prices[0, -2])}"

    exogDict = greeksManager.getGreeksHistoryDict() # current day is already trimmed from the exogs

    assert np.isclose(
        exogDict["inst_0"]["greek_lag_1"].iloc[-1],
        prices[0, -3]
    ), (
        f"Exog for lag 1 is incorrect. Got {exogDict['inst_0']['greek_lag_1'].iloc[-1]}, "
        f"but expected price from 1 days ago {prices[0, -2]}"
    )

    logReturnsForecaster.fit(
        series=logReturns,
        exog=exogDict
    )

def updateLogReturns():
    global logReturns
    pricesInWindow = prices[:, -(TRAINING_WINDOW_SIZE + 1):]
    logReturnsSoFarNp = np.log(pricesInWindow[:, 1:] / pricesInWindow[:, :-1])
    logReturns = pd.DataFrame(logReturnsSoFarNp.T)
    logReturns.columns = [f"inst_{i}" for i in range(logReturns.shape[1])]

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
    exogDict = greeksManager.getGreeksDict()

    prediction = logReturnsForecaster.predict(
        steps   = steps,
        last_window = logReturns.tail(max(logReturnsForecaster.lags)),
        exog    = exogDict,
        levels  = list(logReturns.columns),
    )

    predictedLogReturns = prediction["pred"].values

    predictedLogReturnsHistory.append(predictedLogReturns)

    return predictedLogReturns

def createGreeksManager():
    laggedPricesPrefix  = "greek_lag_"
    volatilityPrefix    = "greek_volatility_"
    momentumPrefix      = "greek_momentum_"
    skewnessPrefix      = "greek_skewness_"
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
    # skewnessDict = {
    #     f"{skewnessPrefix}{window}" : Skewness(TRAINING_WINDOW_SIZE, prices, window)
    #     for window in SKEWNESS_WINDOWS
    # }

    greeksDict = (
            laggedPricesDict  |
            volatilityDict    |
            momentumDict     |
            # skewnessDict     |
            {
                pricesString : Prices(TRAINING_WINDOW_SIZE, prices)
            }
    )

    gm = GreeksManager(greeksDict)

    for name, greek in gm.greeks.items():
        histGreeks = greek.getGreeksHistory()
        if np.isnan(histGreeks).any():
            print(f"[DEBUG] NaN in {name} history!")

        currGreeks = greek.getGreeks()
        if np.isnan(currGreeks).any():
            print(f"[DEBUG] NaN in {name} current greeks!")

    return gm