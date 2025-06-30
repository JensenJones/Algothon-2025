# If log returns < 0 then short as much as we can.
# If log returns > 0 then long as much as we can.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import keras.models
import numpy as np

from greeks.ProduceGreeksData import createGreeksManager

pricesSoFar = None
positions = np.zeros(50)
isInnit = True
model = keras.models.load_model("./greeks/NN/best_model_from_GreeksNNTuning.keras")
greeksManager = createGreeksManager()

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

    predictedLogReturns = model.predict(greeksData)[0].reshape(1, -1)[0]

    # tradable_indices = [6, 12, 19, 21, 25, 27, 30, 32, 33, 41, 42]
    tradable_indices = [2, 4, 6, 10, 12, 14, 16, 21, 22, 25, 29, 32]
    # tradable_indices = range(50)

    positions = np.zeros_like(predictedLogReturns)

    for i in tradable_indices:
        if predictedLogReturns[i] > 0:
            positions[i] = 33333
        elif predictedLogReturns[i] < 0:
            positions[i] = -33333

    return positions

    return positions