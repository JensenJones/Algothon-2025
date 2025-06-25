import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from greeks.GreeksManager import GreeksManager
from greeks.LaggedPrices import LaggedPrices
from greeks.LogReturns import LogReturns
from greeks.RollingMeans import RollingMeans
from greeks.RsiCalculator import RsiCalculator
from greeks.RsiSingleDirection import RsiSingleDirection

import numpy as np


LAG_START_RANGE = 1
LAG_END_RANGE = 6
ROLLING_MEANS_WINDOW_SIZE = 14
RSIC_WINDOW_SIZE = 14
RSI_LONG_THRESHOLD = 30
RSI_SHORT_THRESHOLD = 70

prices = np.loadtxt("./sourceCode/prices.txt").T

def main():
    pricesSoFar = prices[:, 0:1]
    longRsiC = RsiCalculator(pricesSoFar, RSIC_WINDOW_SIZE)
    shortRsiC = RsiCalculator(pricesSoFar, RSIC_WINDOW_SIZE)

    lagged_prices_greeks = [LaggedPrices(pricesSoFar, lag) for lag in range(LAG_START_RANGE, LAG_END_RANGE)]
    greeks = lagged_prices_greeks + [LogReturns(pricesSoFar),
                                     RollingMeans(pricesSoFar, ROLLING_MEANS_WINDOW_SIZE),
                                     RsiSingleDirection(longRsiC, "long", RSI_LONG_THRESHOLD),
                                     RsiSingleDirection(shortRsiC, "short", RSI_SHORT_THRESHOLD)]
    gm = GreeksManager(greeks)

    produceGreeksData(gm)


def produceGreeksData(gm):
    toLog = {}

    for i in range(1, prices.shape[1]):
        gm.update(prices[:, i:i+1])

        for greek in gm.getGreeks():
            greek_name = greek.__class__.__name__

            if hasattr(greek, 'direction'):
                greek_name += f"_{greek.direction}"

            if hasattr(greek, 'lag'):
                greek_name += f"_Lag={greek.lag}"

            if greek_name not in toLog:
                toLog[greek_name] = []

            toLog[greek_name].append(greek.getGreeks())

    for greekToLog, listOfGreeks in toLog.items():
        np.save(f"./greeks/greeksData/{greekToLog}_750_day_data.npy", np.stack(listOfGreeks))



if __name__ == '__main__':
    main()
