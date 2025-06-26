import sys
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# ============ Improved Tuning Parameters ============
TRAINING_START_DAY = 20
TRAINING_END_DAY = 400  # Reduced to prevent overfitting
CV_START_DAY = 400
CV_END_DAY = 500
TEST_START_DAY = 500

EPOCHS = 100  # Reduced epochs
BATCH_SIZE = 32  # Smaller batch size for stability
LEARNING_RATE = 0.0005  # Lower learning rate
DROPOUT_RATE = 0.4  # Increased dropout
L1_REGULARIZATION = 1e-4  # Increased regularization
L2_REGULARIZATION = 1e-3

# Sequence length for LSTM/GRU models
SEQUENCE_LENGTH = 5  # Reduced sequence length


# ====================================================

class ImprovedPricePredictor:
    def __init__(self):
        self.prices = None
        self.logReturns = None
        self.features = None
        self.scaler_X = None
        self.scaler_y = None
        self.model = None

    def load_data(self):
        """Load and prepare data with better error handling"""
        try:
            self.prices = np.loadtxt("./sourceCode/prices.txt")
            self.logReturns = np.load("./greeks/greeksData/LogReturns_750_day_data.npy")

            # Load features with better organization
            lagged_paths = sorted([
                f for f in glob.glob("./greeks/greeksData/LaggedPrices_Lag=*_750_day_data.npy")
                if "LogReturns" not in f
            ])

            additional_features = [
                "./greeks/greeksData/LaggedPrices_lag=1_750_day_data.npy",
                "./greeks/greeksData/BollingerBandsSingleDirection_focusBand=lower_750_day_data.npy",
                "./greeks/greeksData/BollingerBandsSingleDirection_focusBand=upper_750_day_data.npy",
                "./greeks/greeksData/RollingMeans_750_day_data.npy",
                "./greeks/greeksData/RsiSingleDirection_long_750_day_data.npy",
                "./greeks/greeksData/RsiSingleDirection_short_750_day_data.npy",
                "./greeks/greeksData/Volatility_windowSize=5_750_day_data.npy",
                "./greeks/greeksData/Volatility_windowSize=10_750_day_data.npy",
                "./greeks/greeksData/Volatility_windowSize=20_750_day_data.npy",
                "./greeks/greeksData/Momentum_windowSize=3_750_day_data.npy",
                "./greeks/greeksData/Momentum_windowSize=7_750_day_data.npy",
                "./greeks/greeksData/Momentum_windowSize=14_750_day_data.npy"
            ]

            all_features = lagged_paths + additional_features
            self.features = np.stack([np.load(f) for f in all_features], axis=-1)

            print(f"Data loaded successfully:")
            print(f"  Prices shape: {self.prices.shape}")
            print(f"  Log returns shape: {self.logReturns.shape}")
            print(f"  Features shape: {self.features.shape}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def prepare_data_for_sequence_model(self):
        """Prepare data for LSTM/GRU models with sequences"""
        X_seq = []
        y_seq = []

        # Create sequences
        for i in range(SEQUENCE_LENGTH, len(self.features) - 1):
            X_seq.append(self.features[i - SEQUENCE_LENGTH:i])
            y_seq.append(self.logReturns[i + 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Reshape for LSTM input
        X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], -1)
        print(f"X_seq shape: {X_seq.shape}")
        print(f"y_seq shape: {y_seq.shape}")
        return X_seq, y_seq

    def prepare_data(self, use_robust_scaler=True, use_sequences=False):
        """Improved data preparation with better scaling and validation"""
        if use_sequences:
            X, y = self.prepare_data_for_sequence_model()
            # Adjust indices for sequence data
            train_start = max(0, TRAINING_START_DAY - SEQUENCE_LENGTH)
            train_end = TRAINING_END_DAY - SEQUENCE_LENGTH
            cv_start = CV_START_DAY - SEQUENCE_LENGTH
            cv_end = CV_END_DAY - SEQUENCE_LENGTH
            test_start = TEST_START_DAY - SEQUENCE_LENGTH
        else:
            X = self.features[:-1].reshape(self.features.shape[0] - 1, -1)
            y = self.logReturns[1:]
            train_start = TRAINING_START_DAY
            train_end = TRAINING_END_DAY
            cv_start = CV_START_DAY
            cv_end = CV_END_DAY
            test_start = TEST_START_DAY

        # Clean data - remove NaN and Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Choose scaler
        if use_robust_scaler:
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

        # Prepare training data
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]

        # Additional cleaning for training data
        X_train = np.nan_to_num(X_train, nan = 0.0, posinf = 0.0, neginf = 0.0)
        y_train = np.nan_to_num(y_train, nan = 0.0, posinf = 0.0, neginf = 0.0)

        # Fit scalers on training data only
        if use_sequences:
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            self.scaler_X.fit(X_train_reshaped)
            X_train_scaled = self.scaler_X.transform(X_train_reshaped).reshape(X_train.shape)
        else:
            X_train_scaled = self.scaler_X.fit_transform(X_train)

        y_train_scaled = self.scaler_y.fit_transform(y_train)

        # Clean scaled data
        X_train_scaled = np.nan_to_num(X_train_scaled, nan = 0.0, posinf = 0.0, neginf = 0.0)
        y_train_scaled = np.nan_to_num(y_train_scaled, nan = 0.0, posinf = 0.0, neginf = 0.0)

        # Transform validation and test data
        if use_sequences:
            X_cv = X[cv_start:cv_end]
            X_cv = np.nan_to_num(X_cv, nan=0.0, posinf=0.0, neginf=0.0)
            X_cv_reshaped = X_cv.reshape(-1, X_cv.shape[-1])
            X_cv_scaled = self.scaler_X.transform(X_cv_reshaped).reshape(X_cv.shape)

            X_test = X[test_start:]
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = self.scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
        else:
            X_cv_scaled = self.scaler_X.transform(X[cv_start:cv_end])
            X_test_scaled = self.scaler_X.transform(X[test_start:])

        y_cv_scaled = self.scaler_y.transform(y[cv_start:cv_end])
        y_test_scaled = self.scaler_y.transform(y[test_start:])

        # Final cleaning of all scaled data
        X_cv_scaled = np.nan_to_num(X_cv_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_cv_scaled = np.nan_to_num(y_cv_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for NaN values
        print("Data quality check:")
        print(f"  NaNs in X_train: {np.isnan(X_train_scaled).any()}")
        print(f"  NaNs in y_train: {np.isnan(y_train_scaled).any()}")
        print(f"  Inf values in X_train: {np.isinf(X_train_scaled).any()}")
        print(f"  Inf values in y_train: {np.isinf(y_train_scaled).any()}")
        print(f"  X_train range: [{np.min(X_train_scaled):.3f}, {np.max(X_train_scaled):.3f}]")
        print(f"  y_train range: [{np.min(y_train_scaled):.3f}, {np.max(y_train_scaled):.3f}]")

        return X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled

    def create_feedforwardNN_model(self, input_shape, output_shape):
        model = Sequential([
            tf.keras.Input(shape = input_shape),

            Dense(128, activation = 'relu',
                  kernel_regularizer = l1_l2(l1 = L1_REGULARIZATION, l2 = L2_REGULARIZATION),
                  kernel_initializer = 'he_normal'),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),

            Dense(256, activation = 'relu',
                  kernel_regularizer = l1_l2(l1 = L1_REGULARIZATION, l2 = L2_REGULARIZATION),
                  kernel_initializer = 'he_normal'),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),

            Dense(128, activation = 'relu',
                  kernel_regularizer = l1_l2(l1 = L1_REGULARIZATION, l2 = L2_REGULARIZATION),
                  kernel_initializer = 'he_normal'),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),

            Dense(64, activation = 'relu',
                  kernel_regularizer = l1_l2(l1 = L1_REGULARIZATION, l2 = L2_REGULARIZATION),
                  kernel_initializer = 'he_normal'),
            Dropout(DROPOUT_RATE),

            Dense(output_shape, activation = 'linear',
                  kernel_initializer = 'glorot_normal')
        ])

        return model

    def create_lstm_model(self, input_shape, output_shape):
        model = Sequential([
            tf.keras.Input(shape=input_shape),

            # LSTM layers
            LSTM(128, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),
            LSTM(64, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),

            # Dense layers
            Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
            Dropout(DROPOUT_RATE),
            Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
            Dropout(DROPOUT_RATE),

            # Output layer
            Dense(output_shape, activation='linear')
        ])

        return model

    def create_gru_model(self, input_shape, output_shape):
        """Create GRU model for sequence prediction"""
        model = Sequential([
            tf.keras.Input(shape=input_shape),

            GRU(128, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),
            GRU(64, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),

            Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
            Dropout(DROPOUT_RATE),

            Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)),
            Dropout(DROPOUT_RATE),

            Dense(output_shape, activation='linear')
        ])

        return model

    def train_model(self, model_type='feedforward', use_robust_scaler=True):
        """Train model with improved callbacks and validation"""
        use_sequences = model_type in ['lstm', 'gru']

        X, y, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv_scaled, X_test_scaled, y_test_scaled = \
            self.prepare_data(use_robust_scaler = use_robust_scaler, use_sequences = use_sequences)

        # Determine input and output shapes
        if use_sequences:
            input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        else:
            input_shape = (X_train_scaled.shape[1],)

        output_shape = y_train_scaled.shape[1]

        # Create model based on type
        if model_type == 'feedforward':
            self.model = self.create_feedforwardNN_model(input_shape, output_shape)
        elif model_type == 'lstm':
            self.model = self.create_lstm_model(input_shape, output_shape)
        elif model_type == 'gru':
            self.model = self.create_gru_model(input_shape, output_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compile model with custom optimizer
        optimizer = Adam(learning_rate=LEARNING_RATE)
        self.model.compile(optimizer = optimizer, loss = 'huber', metrics = ['mae'])

        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Reduced patience
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-6  # Minimum change to qualify as improvement
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Less aggressive reduction
                patience=8,  # Reduced patience
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',  # Use .keras format to avoid warning
                monitor='val_loss',
                save_best_only=True,
                verbose=0  # Reduce verbosity
            )
        ]

        # Train model
        print(f"Training {model_type} model...")
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_cv_scaled, y_cv_scaled),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        self.evaluate_model(X_test_scaled, y_test_scaled)

        return history

    def evaluate_model(self, X_test_scaled, y_test_scaled):
        """Comprehensive model evaluation with error handling"""
        try:
            # Make predictions
            y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)

            # Inverse transform
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_test = self.scaler_y.inverse_transform(y_test_scaled)

            # Clean predictions
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate metrics for each instrument
            mse_scores = []
            mae_scores = []

            for i in range(y_test.shape[1]):
                mse = mean_squared_error(y_test[:, i], y_pred[:, i])
                mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                mse_scores.append(mse)
                mae_scores.append(mae)

            print(f"\nModel Evaluation Results:")
            print(f"  Average MSE: {np.mean(mse_scores):.6f}")
            print(f"  Average MAE: {np.mean(mae_scores):.6f}")
            print(f"  MSE Std: {np.std(mse_scores):.6f}")
            print(f"  MAE Std: {np.std(mae_scores):.6f}")

            # Calculate directional accuracy
            y_test_direction = np.sign(y_test)
            y_pred_direction = np.sign(y_pred)
            directional_accuracy = np.mean(y_test_direction == y_pred_direction)

            print(f"  Directional Accuracy: {directional_accuracy:.4f}")

            return {
                'mse_scores': mse_scores,
                'mae_scores': mae_scores,
                'directional_accuracy': directional_accuracy
            }

        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return None

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Main execution
def main():
    predictor = ImprovedPricePredictor()

    predictor.load_data()

    models_to_try = ['feedforward', 'lstm', 'gru']

    for model_type in models_to_try:
        print(f"\n{'=' * 50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'=' * 50}")

        try:
            history = predictor.train_model(model_type=model_type, use_robust_scaler=True)
            predictor.plot_training_history(history)
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
            continue


if __name__ == "__main__":
    main()