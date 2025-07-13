import time

import numpy as np
import pandas as pd
import warnings

from scipy.stats import kurtosis
from skforecast.exceptions import MissingValuesWarning
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor


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


class ExponentialMovingAverage(Greek):
    """
    EMA smoothed over `period`, producing self.history of length self.historyWindowSize.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, period: int = 20):
        super().__init__(historyWindowSize)
        self.period = period
        self.alpha = 2 / (period + 1)

        # Need enough past prices: (self.historyWindowSize + period - 1)
        needed = self.historyWindowSize + period - 1
        init_prices = pricesSoFar[:, -needed:]
        n_inst = init_prices.shape[0]

        # Allocate exactly self.historyWindowSize entries
        emas = np.zeros((n_inst, self.historyWindowSize))
        # Seed: simple average of first `period` prices
        emas[:, 0] = np.mean(init_prices[:, :period], axis=1)

        # Build full EMA history
        for t in range(1, self.historyWindowSize):
            price = init_prices[:, period + t - 1]
            emas[:, t] = self.alpha * price + (1 - self.alpha) * emas[:, t - 1]

        self.history = emas  # shape: (nInst, self.historyWindowSize)
        self.last = emas[:, -1]

    def update(self, newDayPrices: np.ndarray):
        new_ema = self.alpha * newDayPrices + (1 - self.alpha) * self.last
        self.last = new_ema
        self.history = np.hstack((self.history[:, 1:], new_ema[:, np.newaxis]))

    def getGreeks(self):
        return self.last

    def getGreeksHistory(self):
        return self.history


class RelativeStrengthIndex(Greek):
    """
    RSI over a rolling windowSize: measures avg gain vs. loss, history length = self.historyWindowSize.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 14):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize

        # Need enough past prices: (self.historyWindowSize + windowSize)
        needed = self.historyWindowSize + windowSize
        init_prices = pricesSoFar[:, -needed:]
        self.pricesSoFar = init_prices.copy()

        history = []
        # Loop over self.historyWindowSize slots
        for i in range(self.historyWindowSize):
            window = init_prices[:, i : i + windowSize + 1]  # shape (nInst, windowSize+1)
            diffs = np.diff(window, axis=1)
            gains = np.clip(diffs, a_min=0, a_max=None)
            losses = np.clip(-diffs, a_min=0, a_max=None)

            avg_gain = np.mean(gains, axis=1)
            avg_loss = np.mean(losses, axis=1) + 1e-8
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            history.append(rsi)

        self.history = np.stack(history, axis=1)  # shape: (nInst, self.historyWindowSize)

    def update(self, newDayPrices: np.ndarray):
        # Roll the buffer
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        window = self.pricesSoFar[:, -self.windowSize - 1 :]
        diffs = np.diff(window, axis=1)
        gains = np.clip(diffs, a_min=0, a_max=None)
        losses = np.clip(-diffs, a_min=0, a_max=None)

        avg_gain = np.mean(gains, axis=1)
        avg_loss = np.mean(losses, axis=1) + 1e-8
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        self.history = np.hstack((self.history[:, 1:], rsi[:, np.newaxis]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class MovingAverageConvergenceDivergence(Greek):
    """
    MACD = EMA(fast) − EMA(slow). Captures trend shifts.
    """
    def __init__(
        self,
        historyWindowSize: int,
        pricesSoFar: np.ndarray,
        fastPeriod: int = 12,
        slowPeriod: int = 26
    ):
        super().__init__(historyWindowSize)
        self.fastPeriod = fastPeriod
        self.slowPeriod = slowPeriod

        # Give each EMA the full pricesSoFar so they slice their own needed window
        self.fast_ema_obj = ExponentialMovingAverage(
            historyWindowSize + (slowPeriod - fastPeriod),
            pricesSoFar,
            fastPeriod
        )
        self.slow_ema_obj = ExponentialMovingAverage(
            historyWindowSize,
            pricesSoFar,
            slowPeriod
        )

        # Align their histories to the same length
        fast_hist = self.fast_ema_obj.getGreeksHistory()[:, -self.historyWindowSize:]
        slow_hist = self.slow_ema_obj.getGreeksHistory()
        self.history = fast_hist - slow_hist  # (nInst, historyWindowSize)
        self.last = self.history[:, -1]

    def update(self, newDayPrices: np.ndarray):
        # Update both EMAs
        self.fast_ema_obj.update(newDayPrices)
        self.slow_ema_obj.update(newDayPrices)

        # Compute new MACD and roll history
        new_macd = self.fast_ema_obj.getGreeks() - self.slow_ema_obj.getGreeks()
        self.history = np.hstack((self.history[:, 1:], new_macd[:, np.newaxis]))
        self.last = new_macd

    def getGreeks(self):
        return self.last

    def getGreeksHistory(self):
        return self.history


class RateOfChange(Greek):
    """
    Simple return over a lag: (P_t / P_{t−lag}) − 1.
    """
    def __init__(self,
                 historyWindowSize: int,
                 pricesSoFar: np.ndarray,
                 lag: int = 1):
        super().__init__(historyWindowSize)
        self.lag = lag
        # use the base class’s historyWindowSize (which is arg+1)
        needed = self.historyWindowSize + lag
        init = pricesSoFar[:, -needed:]
        self.pricesSoFar = init.copy()

        history = []
        # iterate the full self.historyWindowSize, not the raw argument
        for i in range(self.historyWindowSize):
            past    = init[:, i]
            current = init[:, i + lag]
            roc     = current / past - 1
            history.append(roc)

        self.history = np.stack(history, axis=1)  # now shape (nInst, self.historyWindowSize)

    def update(self, newDayPrices: np.ndarray):
        # roll the buffer exactly as you do elsewhere
        self.pricesSoFar = np.hstack((self.pricesSoFar, newDayPrices.reshape(-1, 1)))
        self.pricesSoFar = self.pricesSoFar[:, 1:]

        # compute the latest ROC
        past    = self.pricesSoFar[:, 0]
        current = self.pricesSoFar[:, -1]
        roc     = current / past - 1

        # roll history forward
        self.history = np.hstack((self.history[:, 1:], roc[:, np.newaxis]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


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


class StochasticOscillator(Greek):
    """
    %K Stochastic Oscillator over a rolling window:
    (P_t - LowestLow) / (HighestHigh - LowestLow) * 100
    """

    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 14):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = historyWindowSize + windowSize
        buf = pricesSoFar[:, -needed:]
        self.buffer = buf.copy()  # shape (nInst, needed)

        history = []
        for i in range(self.historyWindowSize):
            window = self.buffer[:, i: i + windowSize + 1]
            low = np.min(window, axis=1)
            high = np.max(window, axis=1)
            curr = window[:, -1]

            with np.errstate(divide='ignore', invalid='ignore'):
                raw = (curr - low) / (high - low) * 100

            pct_k = np.where(high > low, raw, 50.0)

            history.append(pct_k)
        self.history = np.stack(history, axis=1)  # (nInst, historyWindowSize)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1, 1)))
        self.buffer = self.buffer[:, 1:]
        window = self.buffer[:, -self.windowSize - 1:]
        low, high = np.min(window, axis=1), np.max(window, axis=1)
        curr = window[:, -1]
        pct_k = np.where(high > low, (curr - low) / (high - low) * 100, 50.0)
        self.history = np.hstack((self.history[:, 1:], pct_k[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class RollingSkewness(Greek):
    """
    Skewness of log returns over a rolling window: third moment / std^3.
    """

    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 10):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = historyWindowSize + windowSize
        buf = pricesSoFar[:, -needed:]
        self.buffer = buf.copy()

        history = []
        for i in range(self.historyWindowSize):
            window = self.buffer[:, i: i + windowSize + 1]
            lr = np.log(window[:, 1:] / window[:, :-1])
            mean = lr.mean(axis=1, keepdims=True)
            dev = lr - mean
            m2 = np.mean(dev ** 2, axis=1)
            m3 = np.mean(dev ** 3, axis=1)
            skew = m3 / (m2 ** 1.5 + 1e-12)
            history.append(skew)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1, 1)))
        self.buffer = self.buffer[:, 1:]
        window = self.buffer[:, -self.windowSize - 1:]
        lr = np.log(window[:, 1:] / window[:, :-1])
        mean = lr.mean(axis=1, keepdims=True)
        dev = lr - mean
        m2 = np.mean(dev ** 2, axis=1)
        m3 = np.mean(dev ** 3, axis=1)
        skew = m3 / (m2 ** 1.5 + 1e-12)
        self.history = np.hstack((self.history[:, 1:], skew[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class RollingKurtosis(Greek):
    """
    Excess kurtosis of log returns: (m4/m2^2) - 3.
    """

    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 10):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = historyWindowSize + windowSize
        buf = pricesSoFar[:, -needed:]
        self.buffer = buf.copy()

        history = []
        for i in range(self.historyWindowSize):
            window = self.buffer[:, i: i + windowSize + 1]
            lr = np.log(window[:, 1:] / window[:, :-1])
            mean = lr.mean(axis=1, keepdims=True)
            dev = lr - mean
            m2 = np.mean(dev ** 2, axis=1)
            m4 = np.mean(dev ** 4, axis=1)
            kurt = m4 / (m2 ** 2 + 1e-12) - 3
            history.append(kurt)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1, 1)))
        self.buffer = self.buffer[:, 1:]
        window = self.buffer[:, -self.windowSize - 1:]
        lr = np.log(window[:, 1:] / window[:, :-1])
        mean = lr.mean(axis=1, keepdims=True)
        dev = lr - mean
        m2 = np.mean(dev ** 2, axis=1)
        m4 = np.mean(dev ** 4, axis=1)
        kurt = m4 / (m2 ** 2 + 1e-12) - 3
        self.history = np.hstack((self.history[:, 1:], kurt[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class WeightedMovingAverage(Greek):
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, period: int = 20):
        super().__init__(historyWindowSize)
        self.period = period
        self.weights = np.arange(1, self.period + 1)
        total = self.weights.sum()

        # Corrected: use self.historyWindowSize, not the raw argument
        needed = self.historyWindowSize + self.period - 1
        self.buffer = pricesSoFar[:, -needed:].copy()

        history = []
        for i in range(self.historyWindowSize):
            window = self.buffer[:, i : i + self.period]
            # sanity check
            if window.shape[1] != self.period:
                raise ValueError(
                    f"WeightedMA window size mismatch: expected {self.period}, got {window.shape[1]}"
                )
            wma = (window * self.weights).sum(axis=1) / total
            history.append(wma)

        self.history = np.stack(history, axis=1)


    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1, 1)))
        self.buffer = self.buffer[:, 1:]
        window = self.buffer[:, -self.period:]
        wma = (window * self.weights).sum(axis=1) / self.weights.sum()
        self.history = np.hstack((self.history[:, 1:], wma[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class SimpleMovingAverage(Greek):
    """
    SMA over a fixed period: average of the last `period` prices.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, period: int = 20):
        super().__init__(historyWindowSize)
        self.period = period
        needed = self.historyWindowSize + period - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            window = self.buffer[:, i : i + period]
            sma = window.mean(axis=1)
            history.append(sma)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices[:, None]))[:, 1:]
        window = self.buffer[:, -self.period:]
        sma = window.mean(axis=1)
        self.history = np.hstack((self.history[:, 1:], sma[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class BollingerBands(Greek):
    """
    Bollinger Band “width”: (upper – lower) / middle * 100
    where middle = SMA(period), upper = SMA + k*std, lower = SMA – k*std.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray,
                 period: int = 20, k: float = 2.0):
        super().__init__(historyWindowSize)
        self.period = period
        self.k = k
        needed = self.historyWindowSize + period - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            w = self.buffer[:, i : i + period]
            m = w.mean(axis=1)
            s = w.std(axis=1, ddof=1)
            width = ( (m + k*s) - (m - k*s) ) / m * 100
            history.append(width)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices[:, None]))[:, 1:]
        w = self.buffer[:, -self.period:]
        m = w.mean(axis=1)
        s = w.std(axis=1, ddof=1)
        width = ( (m + self.k*s) - (m - self.k*s) ) / m * 100
        self.history = np.hstack((self.history[:, 1:], width[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class WilliamsR(Greek):
    """
    Williams %R over window: (HighestHigh - Close) / (HighestHigh - LowestLow) * -100.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 14):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            w = self.buffer[:, i : i + windowSize + 1]
            low  = w.min(axis=1)
            high = w.max(axis=1)
            curr = w[:, -1]
            denom = high - low + 1e-12
            wr = (high - curr) / denom * -100
            history.append(wr)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices[:, None]))[:, 1:]
        w = self.buffer[:, -self.windowSize-1:]
        low  = w.min(axis=1)
        high = w.max(axis=1)
        curr = w[:, -1]
        denom = high - low + 1e-12
        wr = (high - curr) / denom * -100
        self.history = np.hstack((self.history[:, 1:], wr[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class PriceAcceleration(Greek):
    """
    Change in momentum: momentum(t) - momentum(t-1).
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 1):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize

        # Build and store the rolling buffer
        needed = self.historyWindowSize + windowSize + 1
        self.buffer = pricesSoFar[:, -needed:].copy()

        # Compute initial history
        history = []
        for i in range(self.historyWindowSize):
            p0 = self.buffer[:, i]
            p1 = self.buffer[:, i + windowSize]
            p2 = self.buffer[:, i + windowSize + 1]
            mom1 = np.log(p1 / p0)
            mom2 = np.log(p2 / p1)
            history.append(mom2 - mom1)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        # Roll the buffer forward
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1, 1)))[:, 1:]

        # Recompute the last two log-momentums
        p0 = self.buffer[:, -self.windowSize - 2]
        p1 = self.buffer[:, -self.windowSize - 1]
        p2 = self.buffer[:, -1]
        mom1 = np.log(p1 / p0)
        mom2 = np.log(p2 / p1)
        accel = mom2 - mom1

        # Roll your history and append
        self.history = np.hstack((self.history[:, 1:], accel[:, np.newaxis]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class DoubleExponentialMovingAverage(Greek):
    def __init__(self, historyWindowSize, pricesSoFar, period=20):
        super().__init__(historyWindowSize)
        self.period = period
        # save the EMA object for updates
        self.ema1 = ExponentialMovingAverage(historyWindowSize, pricesSoFar, period)

        # get its full history
        e1 = self.ema1.getGreeksHistory()  # (nInst, historyWindowSize)
        alpha = 2 / (period + 1)

        # compute EMA-of-EMA manually
        n_inst, H = e1.shape
        e2 = np.zeros_like(e1)
        e2[:,0] = np.mean(e1[:,:period], axis=1)
        for t in range(1, H):
            e2[:,t] = alpha*e1[:,t] + (1-alpha)*e2[:,t-1]
        self.e2_last = e2[:,-1]    # keep last for updates

        # DEMA history
        self.history = 2*e1 - e2

    def update(self, newDayPrices):
        # update EMA1
        self.ema1.update(newDayPrices)
        e1_new = self.ema1.getGreeks()

        # update EMA-of-EMA
        alpha = 2 / (self.period + 1)
        e2_new = alpha*e1_new + (1-alpha)*self.e2_last
        self.e2_last = e2_new

        # roll DEMA history
        new_dema = 2*e1_new - e2_new
        self.history = np.hstack((self.history[:,1:], new_dema[:,None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class TripleExponentialMovingAverage(Greek):
    """
    TEMA = 3*EMA1 − 3*EMA2 + EMA3, where EMA2 is EMA of EMA1, EMA3 is EMA of EMA2.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, period: int = 20):
        super().__init__(historyWindowSize)
        alpha = 2 / (period + 1)

        # first EMA
        ema1 = ExponentialMovingAverage(historyWindowSize, pricesSoFar, period)
        e1 = ema1.getGreeksHistory()

        # EMA of EMA1
        n_inst, H = e1.shape
        e2 = np.zeros_like(e1)
        e2[:, 0] = np.mean(e1[:, :period], axis=1)
        for t in range(1, H):
            e2[:, t] = alpha * e1[:, t] + (1 - alpha) * e2[:, t-1]

        # EMA of EMA2
        e3 = np.zeros_like(e2)
        e3[:, 0] = np.mean(e2[:, :period], axis=1)
        for t in range(1, H):
            e3[:, t] = alpha * e2[:, t] + (1 - alpha) * e3[:, t-1]

        # TEMA history
        self.history = 3*e1 - 3*e2 + e3

        # stash intermediate states
        self.ema1 = ema1
        self.e2_last = e2[:, -1]
        self.e3_last = e3[:, -1]

    def update(self, newDayPrices: np.ndarray):
        # update EMA1
        self.ema1.update(newDayPrices)
        e1_new = self.ema1.getGreeks()

        alpha = 2 / (self.ema1.period + 1)

        # update EMA2
        e2_new = alpha * e1_new + (1 - alpha) * self.e2_last
        self.e2_last = e2_new

        # update EMA3
        e3_new = alpha * e2_new + (1 - alpha) * self.e3_last
        self.e3_last = e3_new

        # roll TEMA history
        new_tema = 3*e1_new - 3*e2_new + e3_new
        self.history = np.hstack((self.history[:, 1:], new_tema[:, None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class CommodityChannelIndex(Greek):
    """
    CCI = (Price − SMA) / (0.015 * mean deviation)
    Measures deviation from the average price.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, period: int = 20):
        super().__init__(historyWindowSize)
        self.period = period
        needed = self.historyWindowSize + period - 1
        self.buf = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            window = self.buf[:, i : i + period]
            sma = window.mean(axis=1)
            dev = np.mean(np.abs(window - sma[:,None]), axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                raw = (window[:, -1] - sma) / (0.015 * dev)
            cci = np.where(dev > 0, raw, 0.0)
            history.append(cci)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buf = np.hstack((self.buf, newDayPrices[:,None]))[:,1:]
        window = self.buf[:, -self.period:]
        sma = window.mean(axis=1)
        dev = np.mean(np.abs(window - sma[:,None]), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = (window[:, -1] - sma) / (0.015 * dev)
        cci = np.where(dev > 0, raw, 0.0)
        self.history = np.hstack((self.history[:,1:], cci[:,None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history


class ChandeMomentumOscillator(Greek):
    """
    CMO = (sum gains − sum losses) / (sum gains + sum losses) * 100.
    A momentum oscillator that treats up/down equally.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, period: int = 14):
        super().__init__(historyWindowSize)
        self.period = period
        needed = self.historyWindowSize + period
        self.buf = pricesSoFar[:, -needed:].copy()

        history = []
        for i in range(self.historyWindowSize):
            window = self.buf[:, i : i + period + 1]
            diffs  = np.diff(window, axis=1)
            gains  = np.sum(np.clip(diffs,  a_min=0, a_max=None), axis=1)
            losses = np.sum(np.clip(-diffs, a_min=0, a_max=None), axis=1)
            denom = gains + losses + 1e-12
            cmo   = (gains - losses) / denom * 100
            history.append(cmo)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buf = np.hstack((self.buf, newDayPrices[:,None]))[:,1:]
        window = self.buf[:, -self.period-1:]
        diffs  = np.diff(window, axis=1)
        gains  = np.sum(np.clip(diffs,  a_min=0, a_max=None), axis=1)
        losses = np.sum(np.clip(-diffs, a_min=0, a_max=None), axis=1)
        denom = gains + losses + 1e-12
        cmo   = (gains - losses) / denom * 100
        self.history = np.hstack((self.history[:,1:], cmo[:,None]))

    def getGreeks(self):
        return self.history[:, -1]

    def getGreeksHistory(self):
        return self.history

class PriceDifference(Greek):
    """
    Difference between current price and price `lag` days ago.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, lag: int = 1):
        super().__init__(historyWindowSize)
        self.lag = lag
        # need historyWindowSize + lag
        needed = self.historyWindowSize + lag
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            p0 = self.buffer[:, i]
            p1 = self.buffer[:, i + lag]
            history.append(p1 - p0)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        p0 = self.buffer[:, -self.lag-1]
        p1 = self.buffer[:, -1]
        diff = p1 - p0
        self.history = np.hstack((self.history[:,1:], diff[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class AbsoluteMomentum(Greek):
    """
    Absolute log-return over a window: |log(P_t / P_{t-windowSize})|
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 1):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            p0 = self.buffer[:, i]
            p1 = self.buffer[:, i + windowSize]
            history.append(np.abs(np.log(p1 / p0)))
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        p0 = self.buffer[:, -self.windowSize-1]
        p1 = self.buffer[:, -1]
        m = np.abs(np.log(p1 / p0))
        self.history = np.hstack((self.history[:,1:], m[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class RollingMin(Greek):
    """
    Rolling minimum of price over windowSize.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            history.append(np.min(win, axis=1))
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        m = np.min(self.buffer[:, -self.windowSize:], axis=1)
        self.history = np.hstack((self.history[:,1:], m[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class RollingMax(Greek):
    """
    Rolling maximum of price over windowSize.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            history.append(np.max(win, axis=1))
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        M = np.max(self.buffer[:, -self.windowSize:], axis=1)
        self.history = np.hstack((self.history[:,1:], M[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class RollingRange(Greek):
    """
    Rolling price range: max - min over windowSize.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            history.append(np.max(win,axis=1) - np.min(win,axis=1))
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        rng = np.max(self.buffer[:,-self.windowSize:],axis=1) - np.min(self.buffer[:,-self.windowSize:],axis=1)
        self.history = np.hstack((self.history[:,1:], rng[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class RollingMedian(Greek):
    """
    Rolling median of price over windowSize.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            history.append(np.median(win, axis=1))
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        med = np.median(self.buffer[:,-self.windowSize:],axis=1)
        self.history = np.hstack((self.history[:,1:], med[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class PercentileRank(Greek):
    """
    Percentile rank of newest price within rolling window (inclusive).
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize + 1]
            curr = win[:,-1]
            rank = np.mean(win <= curr[:,None], axis=1)
            history.append(rank)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        win = self.buffer[:,-self.windowSize-1:]
        curr = win[:,-1]
        rank = np.mean(win <= curr[:,None], axis=1)
        self.history = np.hstack((self.history[:,1:], rank[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class CoefficientOfVariation(Greek):
    """
    Std/mean of price over rolling window.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            mean = np.mean(win, axis=1)
            std = np.std(win, axis=1, ddof=1)
            history.append(std / (mean + 1e-12))
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        win = self.buffer[:,-self.windowSize:]
        mean = np.mean(win, axis=1)
        std = np.std(win, axis=1, ddof=1)
        cov = std / (mean + 1e-12)
        self.history = np.hstack((self.history[:,1:], cov[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class MaxDrawdown(Greek):
    """
    Maximum drawdown (%) over rolling window.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            peak = np.maximum.accumulate(win, axis=1)
            dd = np.max((peak - win)/peak, axis=1)
            history.append(dd)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        win = self.buffer[:,-self.windowSize:]
        peak = np.maximum.accumulate(win, axis=1)
        dd = np.max((peak - win)/peak, axis=1)
        self.history = np.hstack((self.history[:,1:], dd[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history

class UlcerIndex(Greek):
    """
    Ulcer index (root mean square drawdown) over window.
    """
    def __init__(self, historyWindowSize: int, pricesSoFar: np.ndarray, windowSize: int = 5):
        super().__init__(historyWindowSize)
        self.windowSize = windowSize
        needed = self.historyWindowSize + windowSize - 1
        self.buffer = pricesSoFar[:, -needed:].copy()
        history = []
        for i in range(self.historyWindowSize):
            win = self.buffer[:, i : i + windowSize]
            peak = np.maximum.accumulate(win, axis=1)
            dd = (peak - win)/peak
            ui = np.sqrt(np.mean(dd**2, axis=1))
            history.append(ui)
        self.history = np.stack(history, axis=1)

    def update(self, newDayPrices: np.ndarray):
        self.buffer = np.hstack((self.buffer, newDayPrices.reshape(-1,1)))[:,1:]
        win = self.buffer[:,-self.windowSize:]
        peak = np.maximum.accumulate(win, axis=1)
        dd = (peak - win)/peak
        ui = np.sqrt(np.mean(dd**2, axis=1))
        self.history = np.hstack((self.history[:,1:], ui[:,None]))

    def getGreeks(self): return self.history[:, -1]
    def getGreeksHistory(self): return self.history


logReturnsForecaster = ForecasterRecursiveMultiSeries(
    # regressor           = HistGradientBoostingRegressor(random_state=8523,
    #                                                     learning_rate=0.05),
    regressor = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=8523,
            verbosity=0
        ),
    transformer_series  = None,
    transformer_exog    = StandardScaler(),
    lags                = 250,
    window_features     = RollingFeatures(
                                stats           = ['min', 'max'],
                                window_sizes    = 250,
                            ),
)

PRICE_LAGS   = [lag for lag in range(1, 8)]
# WINDOW_SIZES = [2, 3, 4, 5, 7, 10, 13, 16, 20, 25, 30, 35, 40, 50, 75, 100]
WINDOW_SIZES = [window for window in range(2, 51)] + [window for window in range(52, 101, 2)] + [window for window in range (102, 250, 3)]


nInst = 50
positions = np.zeros(nInst)
prices: np.ndarray = None
greeksManager: GreeksManager = None
firstInit = True
logReturns: pd.DataFrame = None
currentDay: int = None

TRAINING_MOD = 50
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

    if day < TRAINING_WINDOW_SIZE + max(PRICE_LAGS + WINDOW_SIZES):
        print(f"Shouldn't be hitting this, prcSoFar.shape = {prcSoFar.shape}")
        return positions

    if day % TRAINING_MOD == 0 and not firstInit:
        fitForecaster()

    if firstInit:
        greeksManager = createGreeksManager(prices)
        updateLogReturns(prices)
        fitForecaster()
        firstInit = False

    else:
        greeksManager.updateGreeks(newDayPrices)
        updateLogReturns(prices)

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
    global logReturns, currentDay

    currentDay = prices.shape[1] - 1
    pricesInWindow = prices[:, -(TRAINING_WINDOW_SIZE + 1):]
    logReturnsSoFarNp = np.log(pricesInWindow[:, 1:] / pricesInWindow[:, :-1])

    index = pd.RangeIndex(start=currentDay - TRAINING_WINDOW_SIZE, stop=currentDay)

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

    futureIndex = pd.RangeIndex(start=currentDay,
                               stop=currentDay + steps)

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

def createGreeksManager(prices = prices, T = TRAINING_WINDOW_SIZE):
    laggedPricesPrefix  = "greek_lag_"
    volatilityPrefix    = "greek_volatility_"
    momentumPrefix      = "greek_momentum_"
    rsiPrefix           = "greek_rsi_"
    emaPrefix           = "greek_ema_"
    macdPrefix          = "greek_macd_"
    rocPrefix           = "greek_roc_"
    stochasticKPrefix   = "greek_stochasticSkewness_"
    wmaPrefix           = "greek_wma_"
    kurtosisPrefix      = "greek_kurtosis_"
    skewnessPrefix      = "greek_skewness_"
    smaPrefix           = "greek_sma_"
    bbPrefix            = "greek_bbwidth_"
    willrPrefix         = "greek_williamsr_"
    accelPrefix         = "greek_acceleration_"
    demaPrefix         = "greek_dema_"
    temaPrefix          = "greek_tema_"
    cciPrefix           = "greek_cci_"
    cmoPrefix           = "greek_cmo_"
    priceDiffPrefix     = "greek_pricediff_"
    absDiffPrefix       = "greek_absdiff_"
    rollMinPrefix       = "greek_rollmin_"
    rollMaxPrefix       = "greek_rollmax_"
    rollRangePrefix     = "greek_rollrange_"
    rollMedianPrefix    = "greek_rollmedian_"
    percRankPrefix      = "greek_percentrank_"
    coefVarPrefix       = "greek_coeffvar_"
    maxDDPrefix         = "greek_maxdrawdown_"
    ulcerPrefix         = "greek_ulcerindex_"
    pricesString        = "greek_price"
    pricesString        = "greek_price"

    laggedPricesDict = {
        f"{laggedPricesPrefix}{lag}": LaggedPrices(T, prices, lag)
        for lag in PRICE_LAGS
    }
    volatilityDict = {
        f"{volatilityPrefix}{window}" : Volatility(T, prices, window)
        for window in WINDOW_SIZES
    }
    momentumDict = {
        f"{momentumPrefix}{window}" : Momentum(T, prices, window)
        for window in WINDOW_SIZES
    }
    rsiDict = {
        f"{rsiPrefix}{window}": RelativeStrengthIndex(T, prices, window)
        for window in WINDOW_SIZES
    }
    emaDict = {
        f"{emaPrefix}{window}": ExponentialMovingAverage(T, prices, window)
        for window in WINDOW_SIZES
    }
    macdDict = {
        f"{macdPrefix}": MovingAverageConvergenceDivergence(T, prices)
    }
    rocDict = {
        f"{rocPrefix}{window}": RateOfChange(T, prices, window)
        for window in WINDOW_SIZES
    }
    stochasticDict = {
        f"{stochasticKPrefix}{window}": StochasticOscillator(T, prices, window)
        for window in WINDOW_SIZES
    }
    skewnessDict = {
        f"{skewnessPrefix}{window}": RollingSkewness(T, prices, window)
        for window in WINDOW_SIZES
    }
    kurtosisDict = {
        f"{kurtosisPrefix}{window}": RollingKurtosis(T, prices, window)
        for window in WINDOW_SIZES
    }
    wmaDict = {
        f"{wmaPrefix}{window}": WeightedMovingAverage(T, prices, window)
        for window in WINDOW_SIZES
    }
    smaDict = {
        f"{smaPrefix}{period}": SimpleMovingAverage(T, prices, period)
        for period in WINDOW_SIZES
    }
    bbDict = {
        f"{bbPrefix}{period}": BollingerBands(T, prices, period, k=2.0)
        for period in WINDOW_SIZES
    }
    willrDict = {
        f"{willrPrefix}{window}": WilliamsR(T, prices, window)
        for window in WINDOW_SIZES
    }
    accelDict = {
        f"{accelPrefix}{window}": PriceAcceleration(T, prices, window)
        for window in WINDOW_SIZES
    }
    demaDict = {
        f"{demaPrefix}{period}": DoubleExponentialMovingAverage(T, prices, period)
        for period in WINDOW_SIZES
    }
    temaDict = {
        f"{temaPrefix}{period}": TripleExponentialMovingAverage(T, prices, period)
        for period in WINDOW_SIZES
    }
    cciDict = {
        f"{cciPrefix}{period}": CommodityChannelIndex(T, prices, period)
        for period in WINDOW_SIZES
    }
    cmoDict = {
        f"{cmoPrefix}{period}": ChandeMomentumOscillator(T, prices, period)
        for period in WINDOW_SIZES
    }
    priceDiffDict    = {f"{priceDiffPrefix}{lag}": PriceDifference(T, prices, lag) for lag in PRICE_LAGS}
    absDiffDict      = {f"{absDiffPrefix}{w}": AbsoluteMomentum(T, prices, w) for w in WINDOW_SIZES}
    rollMinDict      = {f"{rollMinPrefix}{w}": RollingMin(T, prices, w) for w in WINDOW_SIZES}
    rollMaxDict      = {f"{rollMaxPrefix}{w}": RollingMax(T, prices, w) for w in WINDOW_SIZES}
    rollRangeDict    = {f"{rollRangePrefix}{w}": RollingRange(T, prices, w) for w in WINDOW_SIZES}
    rollMedianDict   = {f"{rollMedianPrefix}{w}": RollingMedian(T, prices, w) for w in WINDOW_SIZES}
    percRankDict     = {f"{percRankPrefix}{w}": PercentileRank(T, prices, w) for w in WINDOW_SIZES}
    coefVarDict      = {f"{coefVarPrefix}{w}": CoefficientOfVariation(T, prices, w) for w in WINDOW_SIZES}
    maxDDDict        = {f"{maxDDPrefix}{w}": MaxDrawdown(T, prices, w) for w in WINDOW_SIZES}
    ulcerDict        = {f"{ulcerPrefix}{w}": UlcerIndex(T, prices, w) for w in WINDOW_SIZES}

    greeksDict = (
        laggedPricesDict |
        volatilityDict   |
        momentumDict     |
        rsiDict          |
        emaDict          |
        macdDict         |
        rocDict          |
        stochasticDict   |
        skewnessDict     |
        kurtosisDict     |
        wmaDict          |
        smaDict          |
        bbDict           |
        willrDict        |
        accelDict        |
        demaDict         |
        temaDict         |
        cciDict          |
        cmoDict          |
        priceDiffDict | absDiffDict | rollMinDict | rollMaxDict | rollRangeDict |
        rollMedianDict | percRankDict | coefVarDict | maxDDDict | ulcerDict |
        {
            pricesString : Prices(T, prices)
        }
    )

    gm = GreeksManager(greeksDict)

    return gm

def getGreeksManagerForTesting():
    return greeksManager