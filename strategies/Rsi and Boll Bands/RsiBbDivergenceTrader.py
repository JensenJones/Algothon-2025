import os
import numpy as np
from typing import Optional

class BollingerBandsCalculator:
    """
    Middle band is the standard moving average (SMA) of the instrument price over D days.
    Upper band is the SMA + 2 × standard deviation over D days.
    Lower band is the SMA - 2 × standard deviation over D days.
    """
    def __init__(self, prices: np.ndarray, windowSize: int):
        self.prices = prices.copy()
        self.windowSize = windowSize
        self.queueIndex = 0
        self._init_window_elements()

    def _init_window_elements(self):
        self.windowElements = self.prices[:, -self.windowSize:].copy()
        self.windowSum = np.sum(self.windowElements, axis=1)
        self.sma = self.windowSum / self.windowSize
        self.bollingerBands = np.zeros((self.prices.shape[0], 2))
        self._calc_bands()

    def updateWithNewDay(self, newDayPrices: np.ndarray):
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1, 1)])
        self.windowSum -= self.windowElements[:, self.queueIndex]
        self.windowElements[:, self.queueIndex] = newDayPrices
        self.windowSum += newDayPrices
        self.queueIndex = (self.queueIndex + 1) % self.windowSize
        self.sma = self.windowSum / self.windowSize
        self._calc_bands()

    def _calc_bands(self):
        stdev = np.std(self.windowElements, axis=1)
        self.bollingerBands[:, 1] = self.sma + 2 * stdev  # upper
        self.bollingerBands[:, 0] = self.sma - 2 * stdev  # lower

    def getUpperBands(self) -> np.ndarray:
        return self.bollingerBands[:, 1]

    def getLowerBands(self) -> np.ndarray:
        return self.bollingerBands[:, 0]

    def getSma(self) -> np.ndarray:
        return self.sma

class RsiCalculator:
    """
    Wilders' RSI calculation with exponential smoothing.
    """
    def __init__(self, prices: np.ndarray, windowSize: int):
        self.prices = prices.copy()
        self.windowSize = windowSize
        self.avgGain: Optional[np.ndarray] = None
        self.avgLoss: Optional[np.ndarray] = None
        self.rsi: Optional[np.ndarray] = None
        self._initialize()

    def _initialize(self):
        if self.prices.shape[1] < self.windowSize + 1:
            return
        delta = np.diff(self.prices[:, -self.windowSize - 1:], axis=1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        self.avgGain = np.mean(gain, axis=1)
        self.avgLoss = np.mean(loss, axis=1)
        self._calc_rsi()

    def updateWithNewDay(self, newDayPrices: np.ndarray) -> Optional[np.ndarray]:
        self.prices = np.hstack([self.prices, newDayPrices.reshape(-1,1)])
        if self.prices.shape[1] < self.windowSize + 1:
            return None
        if self.avgGain is None or self.avgLoss is None:
            self._initialize()
            return self.rsi
        prev = self.prices[:, -2]
        delta = newDayPrices - prev
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        self.avgGain = (self.avgGain * (self.windowSize - 1) + gain) / self.windowSize
        self.avgLoss = (self.avgLoss * (self.windowSize - 1) + loss) / self.windowSize
        self._calc_rsi()
        return self.rsi

    def _calc_rsi(self):
        rs = self.avgGain / (self.avgLoss + 1e-10)
        self.rsi = 100 - (100 / (1 + rs))

    def getRsi(self) -> Optional[np.ndarray]:
        return self.rsi

class RsiBollingerBandsTrader:
    def __init__(
        self,
        logFilePath: Optional[str],
        prices: np.ndarray,
        bollinger: BollingerBandsCalculator,
        rsi_calc: RsiCalculator,
        purchaseAlpha: int,
        rsiLongThreshold: int = 30,
        rsiShortThreshold: int = 70,
        lookback: int = 50,
        min_rsi_delta: float = 5.0
    ):
        self.logFilePath = logFilePath
        self.bbc = bollinger
        self.rsiC = rsi_calc
        self.purchaseAlpha = purchaseAlpha
        self.rsiLongThreshold = rsiLongThreshold
        self.rsiShortThreshold = rsiShortThreshold
        self.lookback = lookback
        self.min_rsi_delta = min_rsi_delta

        self.price_history = list(prices.flatten())
        init_rsi = self.rsiC.getRsi()
        self.rsi_history = [float(init_rsi[0])] if init_rsi is not None else []

        if self.logFilePath:
            os.makedirs(os.path.dirname(self.logFilePath), exist_ok=True)
            open(self.logFilePath, 'wb').close()
            self.logAll()

        def updatePosition(self, newDayPrice: np.ndarray, position: np.ndarray) -> np.ndarray:
        price = float(np.atleast_1d(newDayPrice)[0])
        pos = float(np.atleast_1d(position)[0])

        # update histories
        self.price_history.append(price)
        self.bbc.updateWithNewDay(np.array([price]))
        new_rsi = self.rsiC.updateWithNewDay(np.array([price]))
        if new_rsi is not None:
            self.rsi_history.append(float(new_rsi[0]))

        # ensure enough data
        if len(self.price_history) < max(self.bbc.windowSize, self.rsiC.windowSize) or len(self.rsi_history) < self.lookback:
            return np.array([pos])

        upper = self.bbc.getUpperBands()[0]
        lower = self.bbc.getLowerBands()[0]
        curr_rsi = self.rsi_history[-1]

        # divergence gating
        div = self._detect_divergence()
        bull_cond = (price <= lower and curr_rsi < self.rsiLongThreshold)
        bear_cond = (price >= upper and curr_rsi > self.rsiShortThreshold)

        # trade if divergence confirms or fallback on pure BB+RSI
        if (div == 'bull' and bull_cond) or (div == 'bear' and bear_cond) or (div is None and (bull_cond or bear_cond)):
            if price <= lower and curr_rsi < self.rsiLongThreshold:
                pos += self.purchaseAlpha * (self.rsiLongThreshold - curr_rsi) / self.rsiLongThreshold
            elif price >= upper and curr_rsi > self.rsiShortThreshold:
                pos -= self.purchaseAlpha * (curr_rsi - self.rsiShortThreshold) / (100 - self.rsiShortThreshold)

        self.logAll()
        return np.array([pos])

        upper = self.bbc.getUpperBands()[0]
        lower = self.bbc.getLowerBands()[0]
        curr_rsi = self.rsi_history[-1]

        # Entry based on pure BB & RSI signals (temporarily bypassing divergence gating)
        if price <= lower and curr_rsi < self.rsiLongThreshold:
            pos += self.purchaseAlpha * (self.rsiLongThreshold - curr_rsi) / self.rsiLongThreshold
        elif price >= upper and curr_rsi > self.rsiShortThreshold:
            pos -= self.purchaseAlpha * (curr_rsi - self.rsiShortThreshold) / (100 - self.rsiShortThreshold)

        self.logAll()
        return np.array([pos])

    def _detect_divergence(self) -> Optional[str]:
        if len(self.rsi_history) < self.lookback:
            return None
        segment_prices = np.array(self.price_history[-self.lookback:])
        segment_rsis = np.array(self.rsi_history[-self.lookback:])
        minima = [i for i in range(1, len(segment_prices)-1)
                  if segment_prices[i] < segment_prices[i-1] and segment_prices[i] < segment_prices[i+1]]
        maxima = [i for i in range(1, len(segment_prices)-1)
                  if segment_prices[i] > segment_prices[i-1] and segment_prices[i] > segment_prices[i+1]]
        if len(minima) >= 2:
            i1, i2 = minima[-2], minima[-1]
            if segment_prices[i2] < segment_prices[i1] and (segment_rsis[i2] - segment_rsis[i1]) >= self.min_rsi_delta:
                return 'bull'
        if len(maxima) >= 2:
            i1, i2 = maxima[-2], maxima[-1]
            if segment_prices[i2] > segment_prices[i1] and (segment_rsis[i1] - segment_rsis[i2]) >= self.min_rsi_delta:
                return 'bear'
        return None

    def logAll(self):
        if not self.logFilePath:
            return
        lower = self.bbc.getLowerBands()[0]
        upper = self.bbc.getUpperBands()[0]
        sma = self.bbc.getSma()[0]
        rsi_val = self.rsi_history[-1]
        price = self.price_history[-1]
        entry = np.array([[lower, upper, sma, rsi_val, price]])
        with open(self.logFilePath, 'ab') as f:
            np.save(f, entry)

# --- getMyPosition ---

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Initialize traders on first call, then update all positions.
    """
    instrumentCount = prcSoFar.shape[0]
    POS_LIMIT = 10_000
    global_traders = getattr(getMyPosition, 'traders', {})
    global_currentPos = getattr(getMyPosition, 'currentPos', np.zeros(instrumentCount))

    day = prcSoFar.shape[1]
    for idx, params in enumerate(best_params):
        window, alpha, r_low, r_high = map(int, params)
        if idx not in global_traders and day >= window:
            bbc = BollingerBandsCalculator(prcSoFar[idx:idx+1,:], window)
            rsi = RsiCalculator(prcSoFar[idx:idx+1,:], window)
            log_path = f"./logs/instr_{idx}_div.npy"
            trader = RsiBollingerBandsTrader(log_path, prcSoFar[idx:idx+1,:], bbc, rsi, alpha, r_low, r_high)
            global_traders[idx] = trader
    # update
    for idx, trader in global_traders.items():
        price = prcSoFar[idx, -1:].reshape(1,)
        pos = np.array([global_currentPos[idx]])
        global_currentPos[idx] = trader.updatePosition(price, pos)[0]
    # clip
    limits = np.floor(POS_LIMIT / prcSoFar[:,-1]).astype(int)
    global_currentPos = np.clip(global_currentPos, -limits, limits)

    # store back
    getMyPosition.traders = global_traders
    getMyPosition.currentPos = global_currentPos
    return global_currentPos

# Load best_params & passed instruments
best_params = np.load('./strategies/Rsi and Boll Bands/best_params_per_instrument.npy')
passed = np.load('./strategies/Rsi and Boll Bands/passedTestValidation.npy')
