import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import display
from skforecast.plot import set_dark_theme
from tqdm import tqdm

import sklearn
import skforecast
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold,
    OneStepAheadFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
    backtesting_forecaster_multiseries,
    bayesian_search_forecaster_multiseries
)
from skforecast.preprocessing import RollingFeatures, series_long_to_dict, exog_long_to_dict
from skforecast.exceptions import OneStepAheadValidationWarning

import warnings

from greeks.GreeksCorrelationChecker import pricesFilePath

warnings.filterwarnings("ignore", category=ResourceWarning)


colourOrangeBold = "\033[1m\033[38;5;208m"
colourReset = "\033[0m"

print(f"{colourOrangeBold}Version skforecast: {skforecast.__version__}{colourReset}")
print(f"{colourOrangeBold}Version scikit-learn: {sklearn.__version__}{colourReset}")
print(f"{colourOrangeBold}Version pandas: {pd.__version__}{colourReset}")
print(f"{colourOrangeBold}Version numpy: {np.__version__}{colourReset}")

# =======================================================
TRAIN_START = 20
TRAIN_END = 600
VAL_END = 675

# =======================================================
# =======================================================

def main():
    data = getData()

    exog_dict = getExog()

    dataPreProcessing(data)

    data_train, data_val, data_test = splitData(data)
    exog_train, exog_val, exog_test = splitExogDict(exog_dict)

    set_dark_theme()
    # plotInstruments(data, 6)

    # uni_series_mae, predictions = trainAndBacktestPerInstrumentForecast(data, data_train, data_val)

    # multi_series_mae, predictions_ms = trainAndBacktestGlobalModel(data, data_train, data_val, exog_dict, exog_train, exog_val)

    # compareModels(multi_series_mae, uni_series_mae)

    # plotForecasts(data, predictions_ms, "inst_5")
    # plotForecasts(data, predictions, "inst_5")

    parameterOptimisationGlobalModel(data, exog_dict)


def search_space(trial):
    return {
        'lags': trial.suggest_categorical('lags', [7, 14, 20]),
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
    }


def parameterOptimisationGlobalModel(data, exog_dict):
    warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)

    window_features = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=7)

    exog_train_val = {
        k: v.iloc[TRAIN_START:VAL_END].copy().reset_index(drop=True)
        for k, v in exog_dict.items()
    }

    exog_long = pd.concat(
        [df.assign(level=inst) for inst, df in exog_train_val.items()],
        ignore_index=True
    )

    forecaster_ms = ForecasterRecursiveMultiSeries(
        regressor=HistGradientBoostingRegressor(random_state=123),
        lags=14,
        window_features=window_features,
        transformer_series=StandardScaler(),
        encoding='ordinal'
    )

    cv_search = OneStepAheadFold(initial_train_size=VAL_END - TRAIN_START)

    results_bayesian_ms, _ = bayesian_search_forecaster_multiseries(
        forecaster=forecaster_ms,
        series=data.loc[TRAIN_START:VAL_END, :],
        exog=exog_long,
        levels=None,
        cv=cv_search,
        search_space=search_space,
        n_trials=20,
        metric='mean_absolute_error',
        show_progress=True
    )

    print(f"{colourOrangeBold}Best configuration:{colourReset}")
    print(results_bayesian_ms.iloc[0])
    return results_bayesian_ms


def plotForecasts(data, predictions, instrument="inst_0"):
    plt.figure(figsize=(10, 7))

    actual = data[instrument].iloc[-len(predictions):]
    predicted = predictions[instrument]

    actual.plot(label="Actual")
    predicted.plot(label="Predicted")

    plt.legend()
    plt.title(f"Forecast vs Actual for {instrument}")
    plt.show()


def trainAndBacktestGlobalModel(data, data_train, data_val, exog_dict, exog_train, exog_val):
    items = list(data.columns)

    exog_train_val = {
        k: pd.concat([exog_train[k], exog_val[k]], ignore_index=True)
        for k in exog_dict
    }

    series_train_val = data.loc[TRAIN_START:VAL_END - 1].copy()

    # Define forecaster
    window_features = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=7)
    forecaster_ms = ForecasterRecursiveMultiSeries(
        regressor=HistGradientBoostingRegressor(random_state=8523),
        lags=20,
        encoding='ordinal',
        transformer_series=StandardScaler(),
        window_features=window_features,
    )

    # Backtesting forecaster for all items
    cv = TimeSeriesFold(
        steps=7,
        initial_train_size=VAL_END - TRAIN_START,
        refit=False,
    )

    multi_series_mae, predictions_ms = backtesting_forecaster_multiseries(
        forecaster=forecaster_ms,
        series=series_train_val,
        levels=items,
        exog=exog_train_val,
        cv=cv,
        metric='mean_absolute_error',
    )

    # Results
    print(f"{colourOrangeBold}========================{colourReset}")
    display(multi_series_mae.head(5))
    display(predictions_ms.head(5))
    print(f"{colourOrangeBold}========================{colourReset}\n\n")

    predictions_ms = predictions_ms.reset_index()
    return multi_series_mae, predictions_ms.pivot(index='index', columns='level', values='pred')


def trainAndBacktestPerInstrumentForecast(data, data_train, data_val):
    items = []
    mae_values = []
    predictions = {}

    for i, item in enumerate(tqdm(data.columns)):
        # Define forecaster
        window_features = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=7)

        forecaster = ForecasterRecursive(
            regressor=HistGradientBoostingRegressor(random_state=8523),
            lags=20,
            window_features=window_features
        )

        # Backtesting forecaster
        cv = TimeSeriesFold(
            steps=7,
            initial_train_size=len(data_train) + len(data_val),
            refit=False,
        )

        metric, preds = backtesting_forecaster(
            forecaster=forecaster,
            y=data[item],
            cv=cv,
            metric='mean_absolute_error',
            show_progress=False
        )

        items.append(item)
        mae_values.append(metric.at[0, 'mean_absolute_error'])
        predictions[item] = preds

    # Results
    uni_series_mae = pd.Series(
        data=mae_values,
        index=items,
        name='uni_series_mae'
    )

    print(f"{colourOrangeBold}========================{colourReset}")
    print(uni_series_mae.head())
    print(f"{colourOrangeBold}========================{colourReset}\n\n")

    return uni_series_mae, predictions


def compareModels(multi_series_mae, uni_series_mae):
    # Difference of backtesting metric for each item
    multi_series_mae = multi_series_mae.set_index('levels')
    multi_series_mae.columns = ['multi_series_mae']
    results = pd.concat((uni_series_mae, multi_series_mae), axis=1)
    results['improvement'] = results.eval('uni_series_mae - multi_series_mae')
    results['improvement_(%)'] = 100 * results.eval('(uni_series_mae - multi_series_mae) / uni_series_mae')
    results = results.round(2)
    results.style.bar(subset=['improvement_(%)'], align='mid', color=['#d65f5f', '#5fba7d'])

    # Average improvement for all items
    print(results[['improvement', 'improvement_(%)']].agg(['mean', 'min', 'max']))

    # Number of series with positive and negative improvement
    print(pd.Series(np.where(results['improvement_(%)'] < 0, 'negative', 'positive')).value_counts())


def plotInstruments(data, instrumentCount=4):
    fig, axs = plt.subplots(instrumentCount, 1, figsize=(7, 5), sharex=True)
    data.iloc[:, :instrumentCount].plot(
        legend=True,
        subplots=True,
        title='First 4 Instruments Prices Over Time',
        ax=axs,
        linewidth=1
    )
    # Add vertical lines at training and validation split
    for ax in axs:
        ax.axvline(x=TRAIN_END, color='white', linestyle='--', linewidth=1.5)
        ax.axvline(x=VAL_END, color='white', linestyle='--', linewidth=1.5)
    fig.tight_layout()
    plt.show()


def splitData(data):
    data_train = data.loc[TRAIN_START:TRAIN_END - 1].copy()
    data_val   = data.loc[TRAIN_END:VAL_END - 1].copy()
    data_test  = data.loc[VAL_END:].copy()


    print(f"Train days      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Validation days : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
    print(f"Test days       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    return (data_train, data_val, data_test)


def splitExogDict(exog_dict):
    exog_train = {}
    exog_val   = {}
    exog_test  = {}

    for inst, df in exog_dict.items():
        exog_train[inst] = df.iloc[TRAIN_START:TRAIN_END].copy()
        exog_val[inst]   = df.iloc[TRAIN_END:VAL_END].copy()
        exog_test[inst]  = df.iloc[VAL_END:].copy()

    return exog_train, exog_val, exog_test


def dataPreProcessing(data):
    data.index.name = 'day'
    data.columns = [f"inst_{i}" for i in range(data.shape[1])]


def getExog():
    greeksFilePaths = sorted(glob.glob("./greeks/greeksData/*.npy"))
    feature_names = [os.path.splitext(os.path.basename(f))[0] for f in greeksFilePaths]

    # Load and stack -> shape (750, 50, N_features)
    exog_array = np.stack([np.load(f) for f in greeksFilePaths], axis=-1)

    # Split per instrument
    exog_dict = {
        f"inst_{i}": pd.DataFrame(exog_array[:, i, :], columns=feature_names)
        for i in range(exog_array.shape[1])
    }

    print(f"{colourOrangeBold}Built exog_dict with {len(exog_dict)} instruments, each shape {exog_dict['inst_0'].shape}{colourReset}")
    print("Features:", feature_names)
    return exog_dict


def getData():
    data = pd.read_csv("./sourceCode/prices.txt", sep=r'\s+', header=None)
    display(data)
    print(f"{colourOrangeBold}Shape: {data.shape}{colourReset}")
    return data


if __name__ == '__main__':
    main()