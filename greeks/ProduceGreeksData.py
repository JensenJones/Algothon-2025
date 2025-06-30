import sys
import os

from greeks.GreekGeneratingClasses.BollingerBandsCalculator import BollingerBandsCalculator
from greeks.GreekGeneratingClasses.BollingerBandsSingleDirection import BollingerBandsSingleDirection
from greeks.GreekGeneratingClasses.Momentum import Momentum
from greeks.GreekGeneratingClasses.Volatility import Volatility

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from greeks.GreeksManager import GreeksManager
from greeks.GreekGeneratingClasses.LaggedPrices import LaggedPrices
from greeks.GreekGeneratingClasses.LogReturns import LogReturns
from greeks.GreekGeneratingClasses.RollingMeans import RollingMeans
from greeks.GreekGeneratingClasses.RsiCalculator import RsiCalculator
from greeks.GreekGeneratingClasses.RsiSingleDirection import RsiSingleDirection

import numpy as np

LAG_START_RANGE = 1
LAG_END_RANGE = 6
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]
ROLLING_MEANS_WINDOW_SIZE = 14
RSIC_WINDOW_SIZE = 14
RSI_LONG_THRESHOLD = 30
RSI_SHORT_THRESHOLD = 70
BB_WINDOW_SIZE = 20

prices = np.loadtxt("./sourceCode/prices.txt").T


def main():
    gm = createGreeksManager()

    produceGreeksData(gm)


def createGreeksManager():
    pricesSoFar = prices[:, 0:1]
    longRsiC = RsiCalculator(pricesSoFar, RSIC_WINDOW_SIZE)
    shortRsiC = RsiCalculator(pricesSoFar, RSIC_WINDOW_SIZE)
    lowerBbc = BollingerBandsCalculator(pricesSoFar, BB_WINDOW_SIZE)
    upperBbc = BollingerBandsCalculator(pricesSoFar, BB_WINDOW_SIZE)

    def lowerBbComp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a < b).astype(int)

    def upperBbComp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a > b).astype(int)

    lagged_prices_greeks = [LaggedPrices(pricesSoFar, lag) for lag in range(LAG_START_RANGE, LAG_END_RANGE)]
    vol_greeks = [Volatility(pricesSoFar, window) for window in VOL_WINDOWS]
    momentum_greeks = [Momentum(pricesSoFar, window) for window in MOMENTUM_WINDOWS]
    greeks = (
            lagged_prices_greeks +
            vol_greeks +
            momentum_greeks +
            [
                LogReturns(pricesSoFar),
                RollingMeans(pricesSoFar, ROLLING_MEANS_WINDOW_SIZE),
                RsiSingleDirection(longRsiC, "long", RSI_LONG_THRESHOLD),
                RsiSingleDirection(shortRsiC, "short", RSI_SHORT_THRESHOLD),
                BollingerBandsSingleDirection(pricesSoFar, lowerBbc, "lower", lowerBbComp),
                BollingerBandsSingleDirection(pricesSoFar, upperBbc, "upper", upperBbComp)
            ])
    gm = GreeksManager(greeks)
    return gm


def produceGreeksData(gm):
    toLog = {}
    addToLog(gm, toLog)

    for i in range(1, prices.shape[1]):
        gm.update(prices[:, i:i + 1])
        addToLog(gm, toLog)

    for greekToLog, listOfGreeks in toLog.items():
        print(f"Appending greeks of name {greekToLog}\nShape: {np.stack(listOfGreeks).shape}\n")
        np.save(f"./greeks/greeksData/{greekToLog}_750_day_data.npy", np.stack(listOfGreeks))


def addToLog(gm, toLog):
    for greek in gm.getGreeksList():
        greek_name = greek.__class__.__name__

        if hasattr(greek, 'direction'):
            greek_name += f"_{greek.direction}"

        if hasattr(greek, 'lag'):
            greek_name += f"_Lag={greek.lag}"

        if hasattr(greek, 'focusBand'):
            greek_name += f"_focusBand={greek.focusBand}"

        if hasattr(greek, 'windowSize'):
            greek_name += f"_windowSize={greek.windowSize}"

        if greek_name not in toLog:
            toLog[greek_name] = []

        assert greek.getGreeksList().shape[0] == 50, f"{greek_name} has shape = {greek.getGreeksList().shape}"
        toLog[greek_name].append(greek.getGreeksList())


if __name__ == '__main__':
    main()
