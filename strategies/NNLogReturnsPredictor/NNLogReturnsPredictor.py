# If log returns < 0 then short as much as we can.
# If log returns > 0 then long as much as we can.
import joblib
import keras.models
import numpy as np

from greeks.ProduceGreeksData import createGreeksManager

pricesSoFar = None
positions = np.zeros(50)
isInnit = True
model = keras.models.load_model("./greeks/NN/best_model_from_GreeksNNTuning.keras")
greeksManager = createGreeksManager()
scaler_y = joblib.load("./greeks/NN/scaler_y_NN_LogReturns.gz")

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global positions

    day = prcSoFar.shape[1]

    if day > 20:
        positions = updatePositions(prcSoFar[:, -1])

    if day % 50 == 0:
        print(f"Day {day}")

    return positions

def updatePositions(newDayPrices) -> np.ndarray:
    greeksManager.update(newDayPrices.reshape(-1, 1))
    greeksData = greeksManager.getGreeks().T
    greeksData = np.concatenate([greeksData, newDayPrices.reshape(-1, 1)], axis=1)

    greeksData = greeksData.flatten().reshape(1, -1)

    predictedLogReturns = scaler_y.inverse_transform(model.predict(greeksData)[0].reshape(1, -1))[0]

    # print("PredictedLogReturns:")
    # print(predictedLogReturns)
    #
    positions = np.where(predictedLogReturns > 0.0001, 10000, 0)
    positions = np.where(predictedLogReturns < -0.0001, -10000, 0)
    # positions = predictedLogReturns * 500000.0

    return positions