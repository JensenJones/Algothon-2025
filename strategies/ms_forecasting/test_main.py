from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from strategies.ms_forecasting.main import createGreeksManager
from strategies.ms_forecasting.main import TRAINING_WINDOW_SIZE
from strategies.ms_forecasting.main import logReturns
from strategies.ms_forecasting.main import PRICE_LAGS
from strategies.ms_forecasting.main import VOL_WINDOWS
from strategies.ms_forecasting.main import MOMENTUM_WINDOWS


class Test(TestCase):
    # TODO check each daily getGreeksDict produces correct data ----------------- DONE
    # TODO check that getGreeksHistory produces correct data
    # TODO Check that log returns are correctly calculated each day ------------- DONE
    # IF THEY ARE WRONG, CHECK IF GETTING THE DATA OUT MANUALLY FROM THE DICT OF GREEK NAME TO GREEK PRODUCES THE CORRECT DATA

    gmDailyExogDicts: list[dict[str, pd.DataFrame]] = []
    actualDailyExogDicts: list[dict[str, pd.DataFrame]] = []

    gmDailyGetGreeksFromMemberVarNpFormat: list[dict[str, np.ndarray]] = []
    actualDailyGreeksToCompareNpFormat: list[dict[str, np.ndarray]] = []

    def setUp(self):
        super().setUp()
        prices = np.loadtxt("./sourceCode/1000Prices.txt").T
        assert prices.shape == (50, 1000), "Shape need to be transposed"

        greeksManager = createGreeksManager(prices[:, :751])
        index = pd.RangeIndex(start=750, stop=751)
        self.gmDailyExogDicts.append(greeksManager.getGreeksDict(index))
        self.actualDailyExogDicts.append(self.produceDayGetGreeksDictLikeGM(prices[:, :751], index))

        self.gmDailyGetGreeksFromMemberVarNpFormat.append(self.getDailyGreeksFromMemberVar(greeksManager))
        self.actualDailyGreeksToCompareNpFormat.append(self.produceDayGreeksDict(prices[:, :751]))

        for day in range(751, 999):
            greeksManager.updateGreeks(prices[:, day])

            index = pd.RangeIndex(start=day - 1, stop=day)
            self.gmDailyExogDicts.append(greeksManager.getGreeksDict(index))
            self.actualDailyExogDicts.append(self.produceDayGetGreeksDictLikeGM(prices[:, :day + 1], index))

            self.gmDailyGetGreeksFromMemberVarNpFormat.append(self.getDailyGreeksFromMemberVar(greeksManager))
            self.actualDailyGreeksToCompareNpFormat.append(self.produceDayGreeksDict(prices[:, :day + 1]))

    def testDailyExogDictsMatchExpectedWhenSkippingGreeksManagerAggregating(self):
        self.assertEqual(
            len(self.gmDailyGetGreeksFromMemberVarNpFormat),
            len(self.actualDailyGreeksToCompareNpFormat),
            "Mismatch in number of days tested, fix the test class"
        )

        for day, (gm_dict, actual_dict) in enumerate(
            zip(self.gmDailyGetGreeksFromMemberVarNpFormat,
                self.actualDailyGreeksToCompareNpFormat),
            start=750
        ):
            self.assertEqual(
                gm_dict.keys(),
                actual_dict.keys(),
                msg="Keys are different in the dictionaries mate"
            )

            for greek_key in gm_dict.keys():
                gm_np = gm_dict[greek_key]
                actual_np = actual_dict[greek_key]

                np.testing.assert_array_equal(gm_np, actual_np,
                                              err_msg=f"Missmatch for the greek {greek_key}, day {day}"
                                             )

    def testDailyExogDictsMatchExpected(self):
        self.assertEqual(
            len(self.gmDailyExogDicts),
            len(self.actualDailyExogDicts),
            "Mismatch in number of days tested, fix the test class"
        )

        for day_idx, (gm_dict, actual_dict) in enumerate(
                zip(self.gmDailyExogDicts, self.actualDailyExogDicts),
                start=750
        ):
            # Keys (inst_0, inst_1, …) should be identical
            self.assertEqual(
                set(gm_dict.keys()),
                set(actual_dict.keys()),
                msg=f"Instrument keys differ on day {day_idx}"
            )

            # Now compare each DataFrame
            for inst_key in gm_dict:
                gm_df = gm_dict[inst_key]
                actual_df = actual_dict[inst_key]

                assert_frame_equal(gm_df, actual_df,
                    check_dtype=True,
                    check_exact=True,
                    check_column_type=True,
                    check_names=True
                )

    def produceDayGetGreeksDictLikeGM(self, prices, index):
        greeksDict = self.produceDayGreeksDict(prices)

        greekNames, greeksData = zip(*greeksDict.items())

        greeksList = [
            greeks.reshape(-1, 1)
            for greeks in greeksData
        ]

        greeksNp = np.concatenate(greeksList, axis=1)

        return {
            f"inst_{inst}": pd.DataFrame(
                greeksNp[inst:inst + 1, :],
                index=index,
                columns=greekNames
            )
            for inst in range(prices.shape[0])
        }

    @staticmethod
    def produceDayGreeksDict(prices) -> dict[str, np.ndarray]:
        lags = {
            f"greek_lag_{lag}": prices[:, -(lag + 1)]
            for lag in PRICE_LAGS
        }

        vols = {
            f"greek_volatility_{window}": np.std(np.log(prices[:, -window:] / prices[:, -(window + 1):-1]), axis=1, ddof=1)
            for window in VOL_WINDOWS
        }

        moms = {
            f"greek_momentum_{window}": np.log(prices[:, -1] / prices[:, -(window + 1)])
            for window in MOMENTUM_WINDOWS
        }

        prices = {"greek_price": prices[:, -1]}

        return lags | vols | moms | prices

    def getDailyGreeksFromMemberVar(self, greeksManager) -> dict[str, np.ndarray]:
        return {
            greekName: greek.getGreeks()
            for greekName, greek in greeksManager.greeks.items()
        }