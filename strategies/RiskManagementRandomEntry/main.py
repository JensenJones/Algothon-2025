import numpy as np

class ATR:
    def __init__(self, initial_prices: np.ndarray, window_size: int):
        self.window_size = window_size

        if initial_prices.shape[1] < window_size + 1:
            raise ValueError(
                f"initial_prices requires at least {window_size + 1} columns"
            )

        self.prices = initial_prices[:, -(window_size + 1):].copy()

        self.atr = np.mean(
            np.abs(np.diff(self.prices, axis=1)),
            axis=1
        )

    def update(self, new_prices: np.ndarray):
        self.prices = np.hstack((
            self.prices[:, 1:],
            new_prices.reshape(-1, 1)
        ))

        self.atr = np.mean(
            np.abs(np.diff(self.prices, axis=1)),
            axis=1
        )

    def getAtr(self) -> np.ndarray:
        """
        Returns:
            Current ATR for each instrument (shape: n_instruments,)
        """
        return self.atr

nInst = 50
window_size = 10
sl_atr_ratio = 3

isInit = True
atrCalc: ATR = None
prices = np.zeros(nInst)
currentPos = np.zeros(nInst)
openPositions = np.zeros(nInst)
stopLosses = np.zeros(nInst)


def generateNewSignalDirection(size: int = nInst) -> np.ndarray:
    return np.random.choice([1, 2], size=size)


def activateStopLosses(currentPos: np.ndarray):
    mask = ((currentPos > 0) & (prices < stopLosses)) | ((currentPos < 0) & (prices > stopLosses))

    currentPos[mask] = 0
    openPositions[mask] = 0

    return currentPos


def placeNewTrades(currentPos: np.ndarray):
    newMask = (currentPos == 0)
    signals = generateNewSignalDirection(nInst)

    openPositions[newMask] = signals[newMask]

    currentPos[newMask] = np.where(signals[newMask] == 1, -9999, 9999)

    factors = np.where(signals == 1, 1 + 0.01, 1 - 0.01)
    stopLosses[newMask] = prices[newMask] * factors[newMask]

    return currentPos


def updateStopLosses(currentPos: np.ndarray, atr: np.ndarray):
    sltr = sl_atr_ratio * atr
    longMask = currentPos == 2
    shortMask = currentPos == 1

    stopLosses[longMask] = np.maximum(
        stopLosses[longMask],
        prices[longMask] - sltr[longMask]
    )
    stopLosses[shortMask] = np.minimum(
        stopLosses[shortMask],
        prices[shortMask] + sltr[shortMask]
    )


def getMyPosition(prcSoFar):
    global isInit, atrCalc, prices

    day = prcSoFar.shape[1]

    if day < 10:
        return np.zeros(nInst)

    if isInit:
        atrCalc = ATR(prcSoFar, 10)
        isInit = False
    else:
        pricesToday = prcSoFar[:, -1]
        atrCalc.update(pricesToday)

    atr = atrCalc.getAtr()

    activateStopLosses(currentPos)
    placeNewTrades(currentPos)
    updateStopLosses(currentPos, atr)

    return currentPos