import sys
import os
import glob

from matplotlib import pyplot as plt
from tensorboard.plugins.scalar.summary import scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import backtesting.backtester as bt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


# ============ Tuning Parameters ============

TRAINING_START_DAY = 14 # once all greeks are valid
TRAINING_END_DAY = 500
CV_START_DAY = 500
CV_END_DAY = 625
TEST_START_DAY = 625

EPOCHS = 500

# ===========================================

prices = None
logReturns = None
features = None
params = None
backtester = None

def main():
    global prices, logReturns, features, params, backtester
    prices = np.loadtxt("./sourceCode/prices.txt")
    logReturns = np.load("./greeks/greeksData/LogReturns_750_day_data.npy")
    lagged_paths = sorted([
        f for f in glob.glob("./greeks/greeksData/LaggedPrices_Lag=*_750_day_data.npy")
        if "LogReturns" not in f
    ])
    greeksFilePaths = (
            lagged_paths +
            [
        # "./greeks/greeksData/LaggedPrices_lag=1_750_day_data.npy",
        "./greeks/greeksData/BollingerBandsSingleDirection_focusBand=lower_750_day_data.npy",
        "./greeks/greeksData/BollingerBandsSingleDirection_focusBand=upper_750_day_data.npy",
        "./greeks/greeksData/RollingMeans_750_day_data.npy",
        "./greeks/greeksData/RsiSingleDirection_long_750_day_data.npy",
        "./greeks/greeksData/RsiSingleDirection_short_750_day_data.npy"
    ])

    features = np.stack([np.load(f) for f in greeksFilePaths], axis=-1)

    print(f"logReturns shape = {logReturns.shape}")
    print(f"prices shape = {prices.shape}")
    print(f"Features shape = {features.shape}")

    # params: bt.Params = bt.parse_command_line_args_as_params(["--path", "./greeks/GreeksNNTuning.py",
    #                                                "--timeline", str(TRAINING_START_DAY), str(TRAINING_END_DAY)])
    # backtester = bt.Backtester(params)

    trainNN()

def trainNN():
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled = setVariables(scaler_X, scaler_y)

    print("NaNs in X_train:", np.isnan(X_train_scaled).any())
    print("NaNs in y_train:", np.isnan(y_train_scaled).any())

    model = createAndTrainModel(X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled)

    plotPredictedLogReturns(model, X_test_scaled, y_test_scaled, scaler_y, "Test data time series prediction of log returns", 1)
    plotPredictedLogReturns(model, scaler_X.transform(X[TRAINING_START_DAY:]), scaler_y.transform(y[TRAINING_START_DAY:]), scaler_y, "All time data time series prediction of log returns", 1)
    plotPredictedPrices(model, scaler_X.transform(X[TRAINING_START_DAY:]), scaler_y.transform(y[TRAINING_START_DAY:]), scaler_y, "All time data time series prediction of prices", 1, prices[1, TRAINING_START_DAY])

def plotPredictedPrices(model, X_scaled, y_scaled, scaler_y, plotTitle, instrument, initial_price):
    predicted_y_test_scaled = model.predict(X_scaled)
    predicted_log_returns = scaler_y.inverse_transform(predicted_y_test_scaled)
    actual_log_returns = scaler_y.inverse_transform(y_scaled)

    predicted_prices = initial_price * np.exp(np.cumsum(predicted_log_returns[:, instrument]))
    actual_prices = initial_price * np.exp(np.cumsum(actual_log_returns[:, instrument]))

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_prices, label="Predicted")
    plt.plot(actual_prices, label="Actual", alpha=0.7)
    plt.title(f"{plotTitle}, instrument = {instrument}")
    plt.xlabel("Day")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def plotPredictedLogReturns(model, X_scaled, y_scaled, scaler_y, plotTitle, instrument):
    predicted_y_test_scaled = model.predict(X_scaled)
    predicted_log_returns = scaler_y.inverse_transform(predicted_y_test_scaled)
    actual_log_returns = scaler_y.inverse_transform(y_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_log_returns[:, instrument], label="Predicted")
    plt.plot(actual_log_returns[:, instrument], label="Actual", alpha=0.7)
    plt.title(f"{plotTitle}, instrument = {instrument}")
    plt.xlabel("Day")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.show()


def createAndTrainModel(X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled):
    model = Sequential([
        tf.keras.Input(shape = (X_train_scaled.shape[1],)),
        Dense(512, activation='relu',  name = "layer1"),
        Dense(1000, activation='relu', name = "layer2"),
        Dense(512, activation='relu',  name = "layer3"),
        Dense(50, activation='linear', name = "outputLayer"),
    ])

    model.compile(optimizer = 'adam', loss = 'huber')
    # early_stop = EarlyStopping(patience = 20, restore_best_weights = True)

    model.fit(X_train_scaled, y_train_scaled,
              validation_data=(X_cv_scaled, y_cv_scaled),
              epochs=EPOCHS,
              # callbacks=[early_stop]
              )

    return model

def setVariables(scaler_x, scaler_y):
    X = features[:-1].reshape(features.shape[0] - 1, -1)  # features from day 0 to 747
    y = logReturns[1:]  # log return from day 1 to 748

    # Fit only on training
    X_train = X[TRAINING_START_DAY:TRAINING_END_DAY]
    y_train = y[TRAINING_START_DAY:TRAINING_END_DAY]

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Only transform CV and test
    X_cv_scaled = scaler_x.transform(X[CV_START_DAY:CV_END_DAY])
    y_cv_scaled = scaler_y.transform(y[CV_START_DAY:CV_END_DAY])

    X_test_scaled = scaler_x.transform(X[TEST_START_DAY:])
    y_test_scaled = scaler_y.transform(y[TEST_START_DAY:])

    return X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled

def getMyPosition():
    pass

if __name__ == "__main__":
    main()