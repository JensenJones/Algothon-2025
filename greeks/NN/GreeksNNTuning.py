import glob
import os
import sys

import joblib
import keras.models
from keras.src.layers import LeakyReLU
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# ============ Tuning Parameters ============

TRAINING_START_DAY = 20 # once all greeks are valid
TRAINING_END_DAY = 500
CV_START_DAY = 500
CV_END_DAY = 625
TEST_START_DAY = 625
TEST_END_DAY = 750

EPOCHS = 1000
BATCH_SIZE = 5 # Number of training examples used to compute one gradient update
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1 # ratio of neurons randomly dropped in training
L1_REGULARIZATION = 1e-8  # Increase if too many irrelevant features
L2_REGULARIZATION = 1e-8  # Increase if overfitting
LEAKY_RELU_NEGATIVE_SLOPE = 0.1

# ===========================================

prices = None
logReturns = None
features = None
params = None
backtester = None

pricesFilePath = "./sourceCode/prices.txt"
modelSaveFilePath = "./greeks/NN/best_model_from_GreeksNNTuning.keras"

def main():
    global prices, logReturns, features, params, backtester, LEAKY_RELU_NEGATIVE_SLOPE
    prices = np.loadtxt(pricesFilePath)
    prices = prices[:, :, np.newaxis]

    greeksFilePaths = [f for f in glob.glob("./greeks/greeksData_750Days/*.npy")]

    features = np.stack([np.load(f) for f in greeksFilePaths], axis=-1)
    features = np.concatenate([features, prices], axis = 2)

    logReturns = np.load("./greeks/greeksData_750Days/LogReturns_lookback=1_750_day_data.npy")

    mask = (logReturns < 0) & (np.isnan(logReturns))
    average_negative = np.mean(logReturns[mask])
    print(f"Average of log returns < 0 = {average_negative}")
    LEAKY_RELU_NEGATIVE_SLOPE = abs(average_negative)

    for featurePath in greeksFilePaths:
        print(f"Using the greek: {featurePath}")
    print(f"Using the greek: {pricesFilePath}")

    print(f"logReturns shape = {logReturns.shape}")
    print(f"prices shape = {prices.shape}")
    print(f"Features shape = {features.shape}")

    trainNN()

def trainNN():
    scaler_X = MinMaxScaler(feature_range=(-10, 10))

    X_flat, y_flat, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled = setVariables(scaler_X)

    print("NaNs in X_train:", np.isnan(X_train_scaled).any())
    print("NaNs in y_train:", np.isnan(y_train_scaled).any())
    print(f"X_train_scaled.shape = {X_train_scaled.shape}")
    print(f"y_train_scaled.shape = {y_train_scaled.shape}")
    print('\n')

    model, history = createAndTrainModel(X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled)

    print('\n')

    plt.title("NN Training Loss Over Epochs")
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("greeks/NN/Training Loss over time.png")

    print(f"prices shape = {prices.shape}")
    print(f"logReturns shape = {logReturns.shape}")

    train_loss, train_mse, train_mae = model.evaluate(scaler_X.transform(X_flat[TRAINING_START_DAY:]),
                                                      y_flat[TRAINING_START_DAY:], verbose=0)
    cv_loss, cv_mse, cv_mae = model.evaluate(scaler_X.transform(X_flat[CV_START_DAY:CV_END_DAY]),
                                             y_flat[CV_START_DAY:CV_END_DAY],
                                             verbose=0)
    test_loss, test_mse, test_mae = model.evaluate(scaler_X.transform(X_flat[TEST_START_DAY:]),
                                                   y_flat[TEST_START_DAY:], verbose=0)

    print(f"Train Loss: {train_loss:.6f}, Train MSE: {train_mse:.6f}, Train MAE: {train_mae:.6f}")
    print(f"CV Loss:    {cv_loss:.6f},    CV MSE: {cv_mse:.6f},    CV MAE: {cv_mae:.6f}")
    print(f"Test Loss:  {test_loss:.6f},  Test MSE: {test_mse:.6f},  Test MAE: {test_mae:.6f}")
    print('\n')

    # Predict and inverse-transform
    y_pred_scaled = model.predict(scaler_X.transform(X_flat[TRAINING_START_DAY:]))
    y_pred = y_pred_scaled

    # Unscaled true labels
    y_true = y_flat[TRAINING_START_DAY:]

    print("\nMSE per instrument on TEST DATA sorted:")
    mse_per_instrument = []
    for i in range(50):
        mse = mean_squared_error(y_true[TEST_START_DAY:, i], y_pred[TEST_START_DAY:, i])
        mse_per_instrument.append(mse)

    sorted_mse = sorted(enumerate(mse_per_instrument), key=lambda x: x[1], reverse=True)
    for i, mse in sorted_mse:
        print(f"Instrument {i:2d}: MSE = {mse:.6f}")
    print('\n')

    print("Predicted log return stats (min, max, std):", y_pred.min(), y_pred.max(), y_pred.std())
    print("Actual log return stats (min, max, std):", y_true.min(), y_true.max(), y_true.std())
    print('\n')

    print(model.summary())

    plotAllPredictedPrices(
        model,
        scaler_X.transform(X_flat),
    )

    plotPredictedLogReturns(
        model,
        scaler_X.transform(X_flat),
        "Predicted vs Actual Log Returns for All Instruments"
    )

def plotPredictedLogReturns(model, X_scaled, title):
    predicted_log_returns_scaled = model.predict(X_scaled)

    predicted_log_returns = predicted_log_returns_scaled

    instruments_per_page = 10
    num_pages = 50 // instruments_per_page

    actualLogReturns = logReturns

    for page in range(num_pages):
        fig, axes = plt.subplots(nrows=instruments_per_page, ncols=1, figsize=(12, 6 * instruments_per_page), sharex=True)
        fig.suptitle(f"{title} (Page {page + 1})", fontsize=16)

        for instrument in range(instruments_per_page):
            i = page * instruments_per_page + instrument
            actual = actualLogReturns[TRAINING_START_DAY:, i]
            predicted = predicted_log_returns[TRAINING_START_DAY:, i]

            ax = axes[instrument]
            ax.plot(predicted, label="Predicted", alpha=0.7)
            ax.plot(actual, label="Actual", alpha=0.7)
            ax.set_ylabel(f"Instr {i}")
            ax.grid(True)

            if instrument == 0:
                ax.legend(loc="upper right", fontsize=10)

        plt.xlabel("Day")
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        filename = f"./greeks/NN/NN_log_returns_page_{page + 1}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")

def plotAllPredictedPrices(model, X_scaled, title = "Predicted vs Actual Prices for All Instruments"):
    print(f"predicting on X_scaled, shape = {X_scaled.shape}")
    predicted_log_returns_scaled = model.predict(X_scaled)

    predicted_log_returns = predicted_log_returns_scaled

    print(f"Predicted log returns shape = {predicted_log_returns.shape}")

    instruments_per_page = 10
    num_pages = 50 // instruments_per_page

    for page in range(num_pages):
        fig, axes = plt.subplots(nrows=instruments_per_page, ncols=1, figsize=(10, 6 * instruments_per_page), sharex=False)
        fig.suptitle(f"{title} (Page {page + 1})", fontsize=16)

        for idx in range(instruments_per_page):
            i = page * instruments_per_page + idx

            instrument_prices = prices[:, i].reshape(-1)
            instrument_predicted_log_returns = predicted_log_returns[:, i].reshape(-1)

            # Predict prices using previous ACTUAL price, not compounding predictions
            predicted_prices = []
            actual_prices = instrument_prices

            for day in range(TRAINING_START_DAY):
                predicted_prices.append(None)
            for day in range(TRAINING_START_DAY, TEST_END_DAY - 1):
                predicted_prices.append(float(instrument_prices[day]) * float(np.exp(instrument_predicted_log_returns[day])))

            predicted_prices_np = np.array(predicted_prices)

            ax = axes[idx]
            ax.plot(predicted_prices_np, label="Predicted")
            ax.plot(actual_prices, label="Actual", alpha=0.7)
            ax.set_ylabel(f"Instr {i}")
            ax.grid(True)

            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        plt.xlabel("Day")
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        filename = f"./greeks/NN/NN_predicted_vs_actual_page_{page + 1}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")

def createAndTrainModel(X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled):
    model = Sequential([
        tf.keras.Input(shape=(X_train_scaled.shape[1],)),

        Dense(128,
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        LeakyReLU(),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        # Dense(256,
        #       kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
        #       kernel_initializer='he_normal'),
        # LeakyReLU(),
        # Dropout(DROPOUT_RATE),
        # BatchNormalization(),

        Dense(512,
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        LeakyReLU(),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        Dense(256,
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        LeakyReLU(),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        Dense(128,
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        LeakyReLU(),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        Dense(50, activation='linear',
              kernel_initializer='glorot_normal')
    ])


    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    callbacks = [
        EarlyStopping( # when cross validation data loss stops improving --> reduce overfitting by stopping early
            monitor='val_loss',
            patience=30,  # Reduced patience
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-6  # Minimum change to qualify as improvement
        ),
        ReduceLROnPlateau( # Decrease the learning rate whenever the cv loss doesn't improve after patience tries
            monitor='val_loss',
            factor=0.3,
            patience=5,
            verbose=1
        ),
        ModelCheckpoint( # For saving the best model
            modelSaveFilePath,  # Use .keras format to avoid warning
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_cv_scaled, y_cv_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )

    model = keras.models.load_model(modelSaveFilePath)

    return model, history

def setVariables(scaler_x):
    X = features[:-1]  # Chop off last day worth of features
    y = logReturns[1:]  # Chop off first day of prices
    # X[i] are now the predictors for price[i] where price[i] is equivalently our prediction of logReturns from our currentDay

    days, instruments, features_dim = X.shape
    X_flat = X.reshape((days, instruments * features_dim))

    X_train = X_flat[TRAINING_START_DAY:TRAINING_END_DAY]
    y_train = y[TRAINING_START_DAY:TRAINING_END_DAY]
    X_cv   = X_flat[CV_START_DAY:CV_END_DAY]
    y_cv   = y[CV_START_DAY:CV_END_DAY]
    X_test = X_flat[TEST_START_DAY:]
    y_test = y[TEST_START_DAY:]

    # Reshape the 3D data of X to 2D so that we can transform it then convert back to 3D
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_cv_reshaped = X_cv.reshape(-1, X_cv.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled_reshaped = scaler_x.fit_transform(X_train_reshaped)
    X_cv_scaled_reshaped = scaler_x.transform(X_cv_reshaped)
    X_test_scaled_reshaped = scaler_x.transform(X_test_reshaped)

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_cv_scaled = X_cv_scaled_reshaped.reshape(X_cv.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

    y_train_scaled = y_train
    y_cv_scaled = y_cv
    y_test_scaled = y_test

    return X_flat, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled

def getMyPosition():
    pass

if __name__ == "__main__":
    main()