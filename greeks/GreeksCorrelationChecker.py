import glob
import os
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ===== File paths =====
pricesFilePath = "./sourceCode/prices.txt"
logReturnsFilePath = "./greeks/greeksData_750Days/LogReturns_lookback=1_750_day_data.npy"

def getData():
    prices = np.loadtxt(pricesFilePath)
    prices = prices[:, :, np.newaxis]

    greeksFilePaths = sorted([f for f in glob.glob("./greeks/greeksData_750Days/*.npy")])  # consistent order

    features_list = []
    feature_names = []

    for f in greeksFilePaths:
        # if "LogReturns" in f:  # skip the target
        #     continue
        features_list.append(np.load(f))
        name = os.path.basename(f).replace("_750_day_data.npy", "")
        feature_names.append(name)

    features = np.stack(features_list, axis=-1)
    features = np.concatenate([features, prices], axis=2)
    feature_names.append("price")

    logReturns = np.load(logReturnsFilePath)

    # Shift features & target to align (predict return at day t using data from day t-1)
    logReturns = logReturns[1:]
    features = features[:-1]

    return features, logReturns, prices, feature_names

def main():
    features, logReturns, _, feature_names = getData()

    print(f"Shape of features = {features.shape}")     # (749, 50, num_features)
    print(f"Shape of logReturns = {logReturns.shape}") # (749, 50)

    features = features[20:]
    logReturns = logReturns[20:]

    num_features = features.shape[-1]
    flattened_log_returns = logReturns.flatten()
    flattened_features = features.reshape(-1, num_features)

    print(f"\n== Linear Correlation ==")
    for f in range(num_features):
        feature_f = features[:, :, f].flatten()
        corr = np.corrcoef(feature_f, flattened_log_returns)[0, 1]
        print(f"{feature_names[f]:<60} correlation with logReturns: {corr:.4f}")

    print(f"\n== Mutual Information ==")
    mi = mutual_info_regression(flattened_features, flattened_log_returns, random_state=42)
    for name, score in zip(feature_names, mi):
        print(f"{name:<60} mutual information with logReturns: {score:.4f}")

    print(f"\n== Random Forest Feature Importance ==")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(flattened_features, flattened_log_returns)
    for name, score in zip(feature_names, rf.feature_importances_):
        print(f"{name:<60} feature importance: {score:.4f}")

    print(f"\n== Linear Regression R² ==")
    lr = LinearRegression()
    lr.fit(flattened_features, flattened_log_returns)
    print(f"Linear model R² score: {lr.score(flattened_features, flattened_log_returns):.4f}")

if __name__ == '__main__':
    main()
