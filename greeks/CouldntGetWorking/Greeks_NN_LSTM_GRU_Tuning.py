import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# ============ Improved Tuning Parameters ============

TRAINING_START_DAY = 14
TRAINING_END_DAY = 500
CV_START_DAY = 500
CV_END_DAY = 625
TEST_START_DAY = 625

# Increased complexity to combat underfitting
EPOCHS = 2000
BATCH_SIZE = 32  # Reduced for better gradient estimates
LEARNING_RATE = 0.001  # Increased for faster learning
DROPOUT_RATE = 0.1  # Reduced to allow more information flow
L1_REGULARIZATION = 1e-6  # Minimal regularization
L2_REGULARIZATION = 1e-5  # Reduced regularization

# New parameters for improved architecture
SEQUENCE_LENGTH = 10  # For LSTM/temporal modeling
USE_ENSEMBLE = True
USE_FEATURE_ENGINEERING = True

# ===========================================

prices = None
logReturns = None
features = None
params = None
backtester = None


def main():
    global prices, logReturns, features, params, backtester
    prices = np.loadtxt("./sourceCode/prices.txt")
    logReturns = np.load("./greeks/greeksData_750Days/LogReturns_lookback=1_750_day_data.npy")

    lagged_paths = sorted([
        f for f in glob.glob("./greeks/greeksData_750Days/LaggedPrices_Lag=*_750_day_data.npy")
        if "LogReturns" not in f
    ])

    greeksFilePaths = (
            lagged_paths +
            [
                "./greeks/greeksData_750Days/LaggedPrices_lag=1_750_day_data.npy",
                "./greeks/greeksData_750Days/BollingerBandsSingleDirection_focusBand=lower_750_day_data.npy",
                "./greeks/greeksData_750Days/BollingerBandsSingleDirection_focusBand=upper_750_day_data.npy",
                "./greeks/greeksData_750Days/RollingMeans_750_day_data.npy",
                "./greeks/greeksData_750Days/RsiSingleDirection_long_750_day_data.npy",
                "./greeks/greeksData_750Days/RsiSingleDirection_short_750_day_data.npy"
            ]
    )

    features = np.stack([np.load(f) for f in greeksFilePaths], axis=-1)

    print(f"logReturns shape = {logReturns.shape}")
    print(f"prices shape = {prices.shape}")
    print(f"Features shape = {features.shape}")

    if USE_ENSEMBLE:
        trainEnsembleNN()
    else:
        trainImprovedNN()


def create_sequences(X, y, sequence_length):
    """Create sequences for temporal modeling"""
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def engineer_features(X):
    """Add engineered features to combat underfitting"""
    if len(X.shape) != 2:
        print(f"Warning: Expected 2D array, got shape {X.shape}")
        return X

    # Polynomial features (degree 2) for non-linear relationships
    X_poly = np.concatenate([X, X ** 2], axis=-1)

    # Cross-instrument statistics
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
    X_max = np.max(X, axis=1, keepdims=True)
    X_min = np.min(X, axis=1, keepdims=True)

    # Interaction features
    X_interactions = np.concatenate([X_mean, X_std, X_max, X_min], axis=-1)

    # Rolling statistics (simplified to avoid indexing issues)
    window = min(5, len(X))
    if len(X) >= window:
        X_rolling_mean = []
        X_rolling_std = []

        for i in range(len(X)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_data = X[start_idx:end_idx]

            X_rolling_mean.append(np.mean(window_data, axis=0))
            X_rolling_std.append(np.std(window_data, axis=0) + 1e-8)

        X_rolling_mean = np.array(X_rolling_mean)
        X_rolling_std = np.array(X_rolling_std)

        return np.concatenate([X_poly, X_interactions, X_rolling_mean, X_rolling_std], axis=-1)

    return np.concatenate([X_poly, X_interactions], axis=-1)


def createLSTMModel(input_shape, output_dim):
    """Create LSTM model for temporal patterns"""
    model = Sequential([
        tf.keras.Input(shape=input_shape),

        LSTM(128, return_sequences=True, dropout=DROPOUT_RATE),
        BatchNormalization(),

        LSTM(64, return_sequences=False, dropout=DROPOUT_RATE),
        BatchNormalization(),

        Dense(256, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        Dense(128, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),

        Dense(output_dim, activation='linear')
    ])

    return model


def createCNNModel(input_shape, output_dim):
    """Create CNN model for pattern recognition"""
    model = Sequential([
        tf.keras.Input(shape=(input_shape,)),

        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),

        Dense(512, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        Dense(256, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),

        Dense(output_dim, activation='linear')
    ])

    return model


def createImprovedDenseModel(input_shape, output_dim):
    """Create improved dense model with more capacity"""
    model = Sequential([
        tf.keras.Input(shape=(input_shape,)),

        # Wider and deeper network
        Dense(512, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(512, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(256, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(256, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(128, activation='relu',
              kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(output_dim, activation='linear',
              kernel_initializer='glorot_normal')
    ])

    return model


def trainEnsembleNN():
    """Train ensemble of different models"""
    # Use RobustScaler for better handling of outliers
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled = setVariables(scaler_X, scaler_y)

    print(f"Original X_train shape: {X_train_scaled.shape}")
    print(f"Original y_train shape: {y_train_scaled.shape}")

    # Engineer features for dense models
    if USE_FEATURE_ENGINEERING:
        print("Engineering features...")
        X_train_eng = engineer_features(X_train_scaled)
        X_cv_eng = engineer_features(X_cv_scaled)
        X_test_eng = engineer_features(X_test_scaled)
        print(f"Engineered X_train shape: {X_train_eng.shape}")
    else:
        X_train_eng = X_train_scaled
        X_cv_eng = X_cv_scaled
        X_test_eng = X_test_scaled

    print("NaNs in X_train_eng:", np.isnan(X_train_eng).any())
    print("NaNs in y_train:", np.isnan(y_train_scaled).any())

    models = []
    histories = []

    # Model 1: Improved Dense Network
    print("Training Dense Model...")
    try:
        model1 = createImprovedDenseModel(X_train_eng.shape[1], y_train_scaled.shape[1])
        optimizer1 = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model1.compile(optimizer=optimizer1, loss='mse', metrics=['mae'])

        callbacks1 = getCallbacks('./greeks/best_dense_model.keras')
        history1 = model1.fit(
            X_train_eng, y_train_scaled,
            validation_data=(X_cv_eng, y_cv_scaled),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks1,
            verbose=1,
            shuffle=False
        )

        models.append(model1)
        histories.append(history1)
        print("Dense model trained successfully!")

    except Exception as e:
        print(f"Error training dense model: {e}")
        return

    # Model 2: LSTM for temporal patterns (if enough data)
    if len(X_train_scaled) >= SEQUENCE_LENGTH:
        print("Training LSTM Model...")
        try:
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
            X_cv_seq, y_cv_seq = create_sequences(X_cv_scaled, y_cv_scaled, SEQUENCE_LENGTH)

            print(f"LSTM input shape: {X_train_seq.shape}")

            model2 = createLSTMModel((SEQUENCE_LENGTH, X_train_scaled.shape[1]), y_train_scaled.shape[1])
            optimizer2 = Adam(learning_rate=LEARNING_RATE / 2, clipnorm=1.0)
            model2.compile(optimizer=optimizer2, loss='mse', metrics=['mae'])

            callbacks2 = getCallbacks('./greeks/best_lstm_model.keras')
            history2 = model2.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_cv_seq, y_cv_seq),
                epochs=EPOCHS // 2,
                batch_size=BATCH_SIZE,
                callbacks=callbacks2,
                verbose=1,
                shuffle=False
            )

            models.append(model2)
            histories.append(history2)
            print("LSTM model trained successfully!")

        except Exception as e:
            print(f"Error training LSTM model: {e}")

    if not models:
        print("No models were successfully trained!")
        return

    # Evaluate ensemble
    print("Evaluating models...")
    evaluateEnsemble(models, X, y, scaler_X, scaler_y)

    # Plot training history
    plotTrainingHistory(histories)


def trainImprovedNN():
    """Train single improved model"""
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled = setVariables(scaler_X, scaler_y)

    print(f"Original shapes - X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")

    # Engineer features
    if USE_FEATURE_ENGINEERING:
        print("Engineering features...")
        X_train_scaled = engineer_features(X_train_scaled)
        X_cv_scaled = engineer_features(X_cv_scaled)
        X_test_scaled = engineer_features(X_test_scaled)
        print(f"After feature engineering - X_train: {X_train_scaled.shape}")

    print("NaNs in X_train:", np.isnan(X_train_scaled).any())
    print("NaNs in y_train:", np.isnan(y_train_scaled).any())

    try:
        model = createImprovedDenseModel(X_train_scaled.shape[1], y_train_scaled.shape[1])

        # Use different optimizers and learning rate scheduling
        optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        callbacks = getCallbacks('./greeks/best_improved_model.keras')

        print("Starting training...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_cv_scaled, y_cv_scaled),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Val MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./greeks/training_history.png')
        plt.show()

        # Evaluate model
        evaluateModel(model, X, y, scaler_X, scaler_y)

    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()


def getCallbacks(model_path):
    """Get improved callbacks"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=50,  # Increased patience for underfitting
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-7
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Less aggressive reduction
            patience=15,  # Increased patience
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]


def evaluateEnsemble(models, X, y, scaler_X, scaler_y):
    """Evaluate ensemble of models"""
    print("\nEvaluating Ensemble...")

    try:
        # Make predictions with each model
        predictions = []

        for i, model in enumerate(models):
            print(f"Making predictions with model {i + 1}...")

            if i == 0:  # Dense model with engineered features
                if USE_FEATURE_ENGINEERING:
                    X_eval = engineer_features(scaler_X.transform(X[TEST_START_DAY:]))
                else:
                    X_eval = scaler_X.transform(X[TEST_START_DAY:])
                pred = model.predict(X_eval, verbose=0)
                predictions.append(pred)

            elif len(models) > 1:  # LSTM model
                X_eval = scaler_X.transform(X[TEST_START_DAY:])
                if len(X_eval) >= SEQUENCE_LENGTH:
                    X_seq, _ = create_sequences(X_eval, y[TEST_START_DAY:], SEQUENCE_LENGTH)
                    pred = model.predict(X_seq, verbose=0)
                    # Pad predictions to match length
                    padding_length = len(X_eval) - len(pred)
                    if padding_length > 0:
                        padding = np.zeros((padding_length, pred.shape[1]))
                        pred = np.vstack([padding, pred])
                    predictions.append(pred)

        if not predictions:
            print("No predictions available!")
            return

        # Ensure all predictions have the same shape
        min_length = min(pred.shape[0] for pred in predictions)
        predictions = [pred[:min_length] for pred in predictions]

        # Average ensemble predictions
        ensemble_pred = np.mean(predictions, axis=0)

        # Calculate metrics
        y_true = scaler_y.transform(y[TEST_START_DAY:TEST_START_DAY + min_length])
        mse = np.mean((ensemble_pred - y_true) ** 2)
        mae = np.mean(np.abs(ensemble_pred - y_true))

        print(f"Ensemble Test MSE: {mse:.6f}")
        print(f"Ensemble Test MAE: {mae:.6f}")

        # Individual model performance
        for i, model in enumerate(models):
            if i == 0:  # Dense model
                if USE_FEATURE_ENGINEERING:
                    X_test_eval = engineer_features(scaler_X.transform(X[TEST_START_DAY:]))
                else:
                    X_test_eval = scaler_X.transform(X[TEST_START_DAY:])
                y_test_eval = scaler_y.transform(y[TEST_START_DAY:])
                test_mse, test_mae = model.evaluate(X_test_eval, y_test_eval, verbose=0)
                print(f"Dense Model - Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")

    except Exception as e:
        print(f"Error in ensemble evaluation: {e}")
        # Fallback to individual model evaluation
        if models:
            evaluateModel(models[0], X, y, scaler_X, scaler_y)


def evaluateModel(model, X, y, scaler_X, scaler_y):
    """Evaluate single model"""
    # Prepare data
    if USE_FEATURE_ENGINEERING:
        X_eval = engineer_features(scaler_X.transform(X))
    else:
        X_eval = scaler_X.transform(X)

    y_eval = scaler_y.transform(y)

    # Evaluate on different splits
    train_mse, train_mae = model.evaluate(X_eval[TRAINING_START_DAY:TRAINING_END_DAY],
                                          y_eval[TRAINING_START_DAY:TRAINING_END_DAY], verbose=0)
    cv_mse, cv_mae = model.evaluate(X_eval[CV_START_DAY:CV_END_DAY],
                                    y_eval[CV_START_DAY:CV_END_DAY], verbose=0)
    test_mse, test_mae = model.evaluate(X_eval[TEST_START_DAY:],
                                        y_eval[TEST_START_DAY:], verbose=0)

    print(f"\nModel Evaluation:")
    print(f"Train MSE: {train_mse:.6f}, Train MAE: {train_mae:.6f}")
    print(f"CV MSE:    {cv_mse:.6f}, CV MAE:    {cv_mae:.6f}")
    print(f"Test MSE:  {test_mse:.6f}, Test MAE:  {test_mae:.6f}")

    # Plot predictions
    plotAllPredictedPrices(
        model,
        X_eval[TRAINING_START_DAY:],
        y_eval[TRAINING_START_DAY:],
        scaler_y,
        prices[TRAINING_START_DAY],
        "Improved Model: Predicted vs Actual Prices"
    )


def plotTrainingHistory(histories):
    """Plot training history for ensemble"""
    plt.figure(figsize=(15, 5))

    for i, history in enumerate(histories):
        plt.subplot(1, 3, i + 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'Model {i + 1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig('./greeks/ensemble_training_history.png')
    plt.show()


def plotAllPredictedPrices(model, X_scaled, y_scaled, scaler_y, initial_prices, title):
    """Plot predictions vs actual prices"""
    predicted_y_scaled = model.predict(X_scaled)
    predicted_log_returns = scaler_y.inverse_transform(predicted_y_scaled)
    actual_log_returns = scaler_y.inverse_transform(y_scaled)

    instruments_per_page = 10
    num_pages = 50 // instruments_per_page

    for page in range(num_pages):
        fig, axes = plt.subplots(nrows=instruments_per_page, ncols=1, figsize=(12, 6 * instruments_per_page))
        fig.suptitle(f"{title} (Page {page + 1})", fontsize=16)

        for idx in range(instruments_per_page):
            i = page * instruments_per_page + idx
            predicted_prices = initial_prices[i] * np.exp(np.cumsum(predicted_log_returns[:, i]))
            actual_prices = initial_prices[i] * np.exp(np.cumsum(actual_log_returns[:, i]))

            ax = axes[idx]
            ax.plot(predicted_prices, label="Predicted", linewidth=2)
            ax.plot(actual_prices, label="Actual", alpha=0.8, linewidth=2)
            ax.set_ylabel(f"Instrument {i}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

        plt.xlabel("Day")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"./greeks/improved_predictions_page_{page + 1}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {filename}")


def setVariables(scaler_x, scaler_y):
    """Prepare variables with improved preprocessing"""
    X = features[:-1].reshape(features.shape[0] - 1, -1)
    y = logReturns[1:]

    # Fit scalers only on training data
    X_train = X[TRAINING_START_DAY:TRAINING_END_DAY]
    y_train = y[TRAINING_START_DAY:TRAINING_END_DAY]

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Transform validation and test sets
    X_cv_scaled = scaler_x.transform(X[CV_START_DAY:CV_END_DAY])
    y_cv_scaled = scaler_y.transform(y[CV_START_DAY:CV_END_DAY])

    X_test_scaled = scaler_x.transform(X[TEST_START_DAY:])
    y_test_scaled = scaler_y.transform(y[TEST_START_DAY:])

    return X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled


def getMyPosition():
    pass


if __name__ == "__main__":
    main()