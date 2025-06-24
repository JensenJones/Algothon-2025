import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import backtesting.backtester as bt

# ============ Tuning Parameters ============

TRAINING_START_DAY = 1
TRAINING_END_DAY = 500
CV_START_DAY = 500
CV_END_DAY = 625
TEST_START_DAY = 625
TEST_END_DAY = 750

# ===========================================

prices = None
features = None
params = None
backtester = None

def main():
    global prices, features, params, backtester

    prices = np.loadtxt("./sourceCode/prices.txt")
    greeksFilePaths = ["./greeks/greeksData/LaggedPrices_750_day_data.npy",
                       "./greeks/greeksData/LogReturns_750_day_data.npy",
                       "./greeks/greeksData/RollingMeans_750_day_data.npy",
                       "./greeks/greeksData/RsiSingleDirection_long_750_day_data.npy",
                       "./greeks/greeksData/RsiSingleDirection_short_750_day_data.npy"]
    features = np.stack([np.load(f) for f in greeksFilePaths], axis=-1)

    print(f"Prices shape = {prices.shape}")
    print(f"Features shape = {features.shape}")

    params = bt.parse_command_line_args_as_params(["--path", "./greeks/GreeksRegressionModel.py",
                                                   "--timeline", str(TRAINING_START_DAY), str(TRAINING_END_DAY)])
    backtester = bt.Backtester(params)

    w = np.zeros((50, 5))
    pass

def getMyPosition():
    pass

if __name__ == "__main__":
    main()