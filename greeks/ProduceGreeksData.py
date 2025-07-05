import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from greeks.GreekGeneratingClasses.Momentum import Momentum
from greeks.GreekGeneratingClasses.Prices import Prices
from greeks.GreekGeneratingClasses.Volatility import Volatility
from greeks.GreeksManager import GreeksManager
from greeks.GreekGeneratingClasses.LaggedPrices import LaggedPrices
# from greeks.GreekGeneratingClasses.LogReturns import LogReturns

import numpy as np

PRICE_LAGS = [1, 2, 3, 4, 5]
VOL_WINDOWS = [5, 10, 20]
MOMENTUM_WINDOWS = [3, 7, 14]
TRAINING_WINDOW_SIZE = 100

prices = np.loadtxt("./sourceCode/prices.txt").T


def main():
    gm = createGreeksManager()
    produceGreeksData(gm)
    print("Produced the data for you cuz")

def createGreeksManager():
    # Dictionary keys match those used in exog in training of the model so that the transformer can work correctly
    # Haven't got this working yet though, I think best approach is to create a new model at the beginning

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

    gm = GreeksManager(greeksDict)

    for name, greek in gm.greeks.items():
        gh = greek.getGreeksHistory()
        if np.isnan(gh).any():
            print(f"[DEBUG] NaN in {name} history!")

    return gm


def produceGreeksData(gm):
    toLog = {}
    addToLog(gm, toLog)

    for i in range(1, prices.shape[1]):
        gm.updateGreeks(prices[:, i:i + 1])
        addToLog(gm, toLog)

    for greekToLog, listOfGreeks in toLog.items():
        print(f"Appending greeks of name {greekToLog}\nShape: {np.stack(listOfGreeks).shape}\n")
        np.save(f"./greeks/greeksData_750Days/{greekToLog}_750_day_data.npy", np.stack(listOfGreeks))

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

        if hasattr(greek, 'lookback'):
            greek_name += f"_lookback={greek.lookback}"

        if greek_name not in toLog:
            toLog[greek_name] = []

        assert greek.getGreeks().shape[0] == 50, f"{greek_name} has shape = {greek.getGreeks().shape}"
        toLog[greek_name].append(greek.getGreeks())

if __name__ == '__main__':
    main()
