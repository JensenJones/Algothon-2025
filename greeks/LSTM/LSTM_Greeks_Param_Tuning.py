# Basing the design on Pedram Jahangiry on Youtube

import glob
import os
import sys
import time

import keras.utils
import numpy as np
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============ Data Split Params ============

TRAINING_START_DAY = 20 # once all greeks are valid (the largest window size out of all the greeks)
TRAINING_END_DAY = 500
CV_START_DAY = 500
CV_END_DAY = 625
TEST_START_DAY = 625

# ============ Timeseries Params ============

SAMPLING_RATE = 1       # Only use every sampling_rate'th data point
SEQUENCE_LENGTH = 20    # Length of each input sequence
DELAY = 1               # Prediction offset (We want to predict the log returns for the next day, hence delay = 1)
BATCH_SIZE = 32         # How many sequences are in one training update step

# =============== LSTM Params ===============

EPOCHS = 40


# ===========================================

pricesFilePath: str = "./sourceCode/prices.txt"
logReturnsFilePath = "./greeks/greeksData_750Days/LogReturns_lookback=1_750_day_data.npy"
modelSaveFilePath = "./greeks/LSTM/Best_LSTM_Produced.keras"
prices = np.loadtxt(pricesFilePath)
logReturns = np.load(logReturnsFilePath)

def main():
    X, y = getData()
    printDataInfo(X, y)

    X_flat, y_flat = flattenData(X, y)

    X_norm, y_norm, X_scaler, y_scaler = normaliseData(X_flat, y_flat)

    print(f"X data has nan values = {np.isnan(X_norm[TRAINING_START_DAY:]).any()}")
    print(f"y data has nan values = {np.isnan(y_norm).any()}")

    nan_indices_X = np.argwhere(np.isnan(X_norm))
    print(f"NaNs in X at indices (day, instrument):\n{len(nan_indices_X)}")

    train_dataset_flattened, cv_dataset_flattened, test_dataset_flattened = createSequenceData(X_norm, y_norm)



    printTrainingDataExampleShape(train_dataset_flattened, "Initial Sequence shape:")

    model, history = createLSTMModel(X_norm,
                                     train_dataset_flattened,
                                     cv_dataset_flattened,
                                     test_dataset_flattened)

    print(model.summary())

    plt.title("NN Training Loss Over Epochs")
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("greeks/LSTM/Training Loss over time.png")

    plotAllPredictedPricesLSTM(model, X_norm, y_scaler)
    plotPredictedLogReturnsLSTM(
        model,
        X_norm,
        y_scaler,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        instruments_per_page=10,
        out_dir="./greeks/LSTM"
    )


def plotAllPredictedPricesLSTM(model, X_norm, y_scaler, sequence_length=SEQUENCE_LENGTH,
                                                        batch_size=BATCH_SIZE,
                                                        instruments_per_page=10):
    # Build sliding windows of sequence_length
    total_days, n_features = X_norm.shape
    num_sequences = total_days - sequence_length + 1
    sequences = np.array([
        X_norm[i : i + sequence_length]
        for i in range(num_sequences)
    ])  # shape = (num_sequences, sequence_length, n_features)

    # Predict scaled log returns
    y_pred_scaled = model.predict(sequences, batch_size=batch_size)
    log_ret_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Reconstruct predicted price paths
    n_instruments = prices.shape[1]
    pred_days = log_ret_pred.shape[0]
    # Initialize an array for predicted prices: one more day than returns
    predicted_prices = np.zeros((pred_days + 1, n_instruments))
    # Start from the first available actual price after the initial window
    predicted_prices[0] = prices[sequence_length - 1]

    for t in range(pred_days):
        predicted_prices[t + 1] = (
            prices[t + sequence_length - 1] * np.exp(log_ret_pred[t])
        )

    # Align actual prices for plotting
    actual_prices = prices[sequence_length - 1 : sequence_length - 1 + pred_days + 1]

    # Paginate and save plots
    num_pages = n_instruments // instruments_per_page
    for page in range(num_pages):
        fig, axes = plt.subplots(
            nrows=instruments_per_page,
            ncols=1,
            figsize=(10, 6 * instruments_per_page),
            sharex=False
        )
        fig.suptitle(f"LSTM Predicted vs Actual Prices (Page {page+1})", fontsize=14)

        for idx in range(instruments_per_page):
            inst = page * instruments_per_page + idx
            ax = axes[idx]
            ax.plot(
                predicted_prices[:, inst],
                label="Predicted",
                linewidth=1
            )
            ax.plot(
                actual_prices[:, inst],
                label="Actual",
                alpha=0.7,
                linewidth=1
            )
            ax.set_ylabel(f"Instr {inst}")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        plt.xlabel("Day index")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = f"./greeks/LSTM/LSTM_pred_vs_actual_page_{page+1}.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

def plotPredictedLogReturnsLSTM(model, X_norm, y_scaler, sequence_length=SEQUENCE_LENGTH,
                                                         batch_size=BATCH_SIZE,
                                                         instruments_per_page=10,
                                                         out_dir="./greeks/LSTM"):
    # 1) Build sliding windows
    total_days, _ = X_norm.shape
    num_sequences = total_days - sequence_length + 1
    sequences = np.stack([X_norm[i : i + sequence_length]
                          for i in range(num_sequences)], axis=0)
    # 2) Predict & inverse-transform
    y_pred_scaled = model.predict(sequences, batch_size=batch_size)
    pred_log = y_scaler.inverse_transform(y_pred_scaled)  # shape = (num_sequences, n_instruments)

    # 3) Align actual
    actual_log = logReturns[sequence_length - 1 :]  # same length as pred_log

    # 4) Paginate & plot
    n_instruments = pred_log.shape[1]
    num_pages = n_instruments // instruments_per_page
    for page in range(num_pages):
        fig, axes = plt.subplots(nrows=instruments_per_page,
                                 ncols=1,
                                 figsize=(12, 6 * instruments_per_page),
                                 sharex=False)
        fig.suptitle(f"Predicted vs Actual Log Returns (Page {page+1})", fontsize=16)

        for idx in range(instruments_per_page):
            inst = page * instruments_per_page + idx
            ax = axes[idx]
            ax.plot(pred_log[:, inst], label="Predicted", alpha=0.7)
            ax.plot(actual_log[:, inst], label="Actual", alpha=0.7)
            ax.set_ylabel(f"Instr {inst}")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=10)

        plt.xlabel("Day Index")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = f"{out_dir}/LSTM_log_returns_page_{page+1}.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

def flattenData(X, y):
    days, instruments, features_dim = X.shape
    X_flat = X.reshape((days, instruments * features_dim))

    print(f"X flattened shape = {X_flat.shape}")
    print(f"y flattened shape = {y.shape}")
    print('\n')

    return X_flat, y

def createLSTMModel(X_norm, train_dataset, cv_dataset, test_dataset):
    inputs = keras.Input(shape=(SEQUENCE_LENGTH, X_norm.shape[-1]))  # (20, 900)

    layer_1 = layers.LSTM(16)(inputs)
    outputs = Dense(50)(layer_1)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(modelSaveFilePath,
                                        save_best_only=True)
    ]

    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=cv_dataset,
        callbacks=callbacks,
        verbose=2,
    )

    model = keras.models.load_model(modelSaveFilePath)

    print(f"TRAIN MAE:            {model.evaluate(train_dataset)[1]:.2f}")
    print(f"CROSS VALIDATION MAE: {model.evaluate(cv_dataset)[1]:.2f}")
    print(f"TEST MAE:             {model.evaluate(test_dataset)[1]:.2f}")

    return model, history

def printTrainingDataExampleShape(trainDataSet, printingTitle):
    print(f"----------{printingTitle}----------")
    for samples, targets in trainDataSet:
        print("Samples shape: ", samples.shape)
        print("targets shape: ", targets.shape)
        print('\n')
        break

def createSequenceData(X, y):
    trainDataSet = keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=SEQUENCE_LENGTH,
        sampling_rate=SAMPLING_RATE,
        batch_size=BATCH_SIZE,
        shuffle=True,                   # This shuffles the different sequences, not the data in the sequences
        start_index=TRAINING_START_DAY,
        end_index=TRAINING_END_DAY,
    )

    cvDataSet = keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=SEQUENCE_LENGTH,
        sampling_rate=SAMPLING_RATE,
        batch_size=BATCH_SIZE,
        shuffle=True,                   # This shuffles the different sequences, not the data in the sequences
        start_index=CV_START_DAY,
        end_index=CV_END_DAY,
    )

    testDataSet = keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=SEQUENCE_LENGTH,
        sampling_rate=SAMPLING_RATE,
        batch_size=BATCH_SIZE,
        shuffle=True,                   # This shuffles the different sequences, not the data in the sequences
        start_index=TEST_START_DAY,
    )

    return trainDataSet, cvDataSet, testDataSet

def normaliseData(X, y):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(X[TRAINING_START_DAY:TRAINING_END_DAY])
    scaler_y.fit(y[TRAINING_START_DAY:TRAINING_END_DAY])

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y

def printDataInfo(X, y):
    print(f"First Training Day (day = {TRAINING_START_DAY}), instrument 0 inspection:")
    print(f"X: {X[TRAINING_START_DAY][0]}")
    print(f"y: {y[TRAINING_START_DAY][0]}")
    print('\n')
    print(f"Number of TRAINING days: {TRAINING_END_DAY - TRAINING_START_DAY}")
    print(f"Number of CROSS VALIDATION days: {CV_END_DAY - CV_START_DAY}")
    print(f"Number of TEST days: {TEST_START_DAY - CV_START_DAY}")
    print('\n')

def getData():
    prices = np.loadtxt(pricesFilePath)
    prices = prices[:, :, np.newaxis]

    greeksFilePaths = [f for f in glob.glob("./greeks/greeksData_750Days/*.npy")]

    features = np.stack([np.load(f) for f in greeksFilePaths], axis=-1)
    features = np.concatenate([features, prices], axis=2)

    logReturns = np.load(logReturnsFilePath)

    for featurePath in greeksFilePaths + [pricesFilePath]:
        print(f"Using the greek: {featurePath}")

    print(f"X (Features) shape = {features.shape}")
    print(f"y (logReturns) shape = {logReturns.shape}")
    print(f"prices shape = {prices.shape}")
    print('\n')

    return features, logReturns

if __name__ == '__main__':
    main()