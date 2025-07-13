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

nInst           = 50
window_size     = 10
momentum_window = 5
sl_atr_ratio    = 2

LONG  = 2
SHORT = 1

isInit = True
atrCalc: ATR = None
prices = np.zeros(nInst)
currentPos = np.zeros(nInst)
openPositionDirections = np.zeros(nInst)
stopLosses = np.zeros(nInst)


def generateNewSignalDirections(size: int = nInst) -> np.ndarray:
    return np.random.choice([SHORT, LONG], size=size)


def activateStopLosses(currentPos: np.ndarray):
    mask = ((openPositionDirections == LONG) & (prices < stopLosses)) | ((openPositionDirections == SHORT) & (prices > stopLosses))

    currentPos[mask] = 0
    openPositionDirections[mask] = 0

    return currentPos


def placeNewTrades(currentPos: np.ndarray, prcSoFar):
    newMask = (currentPos == 0)
    signals = generateNewSignalDirections(nInst)

    openPositionDirections[newMask] = signals[newMask]

    currentPos[newMask] = np.where(signals[newMask] == 1, -9999, 9999)

    sltr = sl_atr_ratio * atrCalc.getAtr()
    stopLosses[newMask & (signals == LONG)] = prices[newMask & (signals == LONG)] - sltr[newMask & (signals == LONG)]
    stopLosses[newMask & (signals == SHORT)] = prices[newMask & (signals == SHORT)] + sltr[newMask & (signals == SHORT)]

    return currentPos, newMask


def updateStopLosses(currentPos: np.ndarray, atr: np.ndarray, needsUpdatingMask: np.ndarray):
    sltr = sl_atr_ratio * atr

    longMask = (currentPos == LONG) & needsUpdatingMask
    shortMask = (currentPos == SHORT) & needsUpdatingMask

    stopLosses[longMask] = np.maximum(
        stopLosses[longMask],
        prices[longMask] - sltr[longMask]
    )
    stopLosses[shortMask] = np.minimum(
        stopLosses[shortMask],
        prices[shortMask] + sltr[shortMask]
    )


def getMyPosition(prcSoFar):
    global isInit, atrCalc, prices, currentPos

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

    currentPos = activateStopLosses(currentPos)
    currentPos, newTradesMask = placeNewTrades(currentPos, prcSoFar)
    updateStopLosses(currentPos, atr, ~newTradesMask)

    return currentPos