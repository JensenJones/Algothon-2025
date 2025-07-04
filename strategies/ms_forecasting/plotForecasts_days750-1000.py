import numpy as np
import pandas as pd
from skforecast.plot import set_dark_theme

import matplotlib.pyplot as plt

set_dark_theme()

def main():
    predictedLogReturns = np.load("./strategies/ms_forecasting/predicted_log_returns_days_750-1000.npy").T # (50, 249)
    print(f"Shape of predictions = {predictedLogReturns.shape}")

    prices = np.loadtxt("./sourceCode/1000Prices.txt").T[:, -250:] # (50, 250)
    print(f"Shape of prices = {prices.shape}")

    actualLogReturns = np.log( prices[:, 1:] / prices[:, :-1] ) # (50, 249)

    instrument = 0

    plt.figure(figsize=(12, 5))
    plt.plot(predictedLogReturns[instrument, :], label="Predicted LogReturn", color='dodgerblue')
    plt.plot(actualLogReturns[instrument, :], label="Actual LogReturn", color='orange')
    plt.title(f"Predicted vs Actual LogReturns for {instrument}")
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()