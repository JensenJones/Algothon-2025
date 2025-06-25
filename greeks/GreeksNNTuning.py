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
    greeksFilePaths = lagged_paths + [
        "./greeks/greeksData/RollingMeans_750_day_data.npy",
        "./greeks/greeksData/RsiSingleDirection_long_750_day_data.npy",
        "./greeks/greeksData/RsiSingleDirection_short_750_day_data.npy"
    ]

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

    X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled = setVariables(scaler_X, scaler_y)

    print("NaNs in X_train:", np.isnan(X_train_scaled).any())
    print("NaNs in y_train:", np.isnan(y_train_scaled).any())

    model = createAndTrainModel(X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled)

    predicted_y_test_scaled = model.predict(X_test_scaled)

    for index in range(3):
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted_y_test_scaled, y_test_scaled, alpha=0.7)
        plt.title("Predicted vs Actual (Test Data)")
        plt.xlabel("Predicted Price (scaled)")
        plt.ylabel("Actual Price (scaled)")
        plt.grid(True)
        plt.show()

def createAndTrainModel(X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled):
    model = Sequential([
        tf.keras.Input(shape = X_train_scaled.shape()),
        Dense(512, activation='relu',  name = "Layer 1"),
        Dense(1000, activation='relu', name = "Layer 2"),
        Dense(512, activation='relu',  name = "Layer 3"),
        Dense(50, activation='linear', name = "Output Layer"),
    ])

    model.compile(optimizer = 'adam', loss = 'huber')
    early_stop = EarlyStopping(patience = 10, restore_best_weights = True)

    model.fit(X_train_scaled, y_train_scaled,
              validation_data=(X_cv_scaled, y_cv_scaled),
              epochs=200,
              batch_size=32,
              callbacks=[early_stop]
              )

    return model

def setVariables(scalar_x, scalar_y):
    X = features
    y = logReturns

    X_train_scaled = scalar_x.fit_transform(X[TRAINING_START_DAY:TRAINING_END_DAY])
    y_train_scaled = scalar_y.fit_transform(y[TRAINING_START_DAY:TRAINING_END_DAY])

    X_cv_scaled = scalar_x.fit_transform(X[CV_START_DAY:CV_END_DAY])
    y_cv_scaled = scalar_y.fit_transform(y[CV_START_DAY:CV_END_DAY])

    X_test_scaled = scalar_x.fit_transform(X[TEST_START_DAY:])
    y_test_scaled = scalar_y.fit_transform(y[TEST_START_DAY:])

    return X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled

def getMyPosition():
    pass

if __name__ == "__main__":
    main()