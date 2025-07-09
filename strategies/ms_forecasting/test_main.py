from unittest import TestCase

import numpy as np
import pandas as pd
from pandas._testing import assert_index_equal
from pandas.testing import assert_frame_equal

from strategies.ms_forecasting.main import createGreeksManager
from strategies.ms_forecasting.main import TRAINING_WINDOW_SIZE
from strategies.ms_forecasting.main import updateLogReturns
from strategies.ms_forecasting.main import PRICE_LAGS
from strategies.ms_forecasting.main import VOL_WINDOWS
from strategies.ms_forecasting.main import MOMENTUM_WINDOWS


class Test(TestCase):
    # TODO check each daily getGreeksDict produces correct data ----------------- DONE
    # TODO check that getGreeksHistory produces correct data
    # TODO Check that log returns are correctly calculated each day

    gmDailyExogDicts: list[dict[str, pd.DataFrame]] = []
    actualDailyExogDicts: list[dict[str, pd.DataFrame]] = []

    gmDailyGetGreeksFromMemberVarNpFormat: list[dict[str, np.ndarray]] = []
    actualDailyGreeksToCompareNpFormat: list[dict[str, np.ndarray]] = []

    gmGreeksHistory = []
    actualGreeksHistory = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        prices = np.loadtxt("./sourceCode/1000Prices.txt").T
        cls.prices = prices
        assert prices.shape == (50, 1000), "Shape need to be transposed"
        cls.logReturns = []

        greeksManager = createGreeksManager(prices[:, :751])

        index = pd.RangeIndex(start=750, stop=751)
        cls.gmDailyExogDicts.append(greeksManager.getGreeksDict(index))
        cls.actualDailyExogDicts.append(cls.produceDayGetGreeksDictLikeGM(prices[:, :751], index))

        cls.gmDailyGetGreeksFromMemberVarNpFormat.append(cls.getDailyGreeksFromMemberVar(greeksManager))
        cls.actualDailyGreeksToCompareNpFormat.append(cls.produceDayGreeksDict(prices[:, :751]))

        historyIndex = pd.RangeIndex(start=750 - TRAINING_WINDOW_SIZE + 1, stop=751)
        cls.gmGreeksHistory.append(greeksManager.getGreeksHistoryDict(historyIndex))
        cls.actualGreeksHistory.append(cls.produceGreeksHistory(prices[:, :751], historyIndex))

        cls.logReturns.append(updateLogReturns(prices[:, :751]))

        for day in range(751, 999):
            greeksManager.updateGreeks(prices[:, day])

            index = pd.RangeIndex(start=day - 1, stop=day)
            cls.gmDailyExogDicts.append(greeksManager.getGreeksDict(index))
            cls.actualDailyExogDicts.append(cls.produceDayGetGreeksDictLikeGM(prices[:, :day + 1], index))

            cls.gmDailyGetGreeksFromMemberVarNpFormat.append(cls.getDailyGreeksFromMemberVar(greeksManager))
            cls.actualDailyGreeksToCompareNpFormat.append(cls.produceDayGreeksDict(prices[:, :day + 1]))

            historyIndex = pd.RangeIndex(start=day - TRAINING_WINDOW_SIZE + 1, stop=day + 1)
            cls.gmGreeksHistory.append(greeksManager.getGreeksHistoryDict(historyIndex))
            cls.actualGreeksHistory.append(cls.produceGreeksHistory(prices[:, :day + 1], historyIndex))

            cls.logReturns.append(updateLogReturns(prices[:, :day + 1]))

    def testLogReturnsIndexAlignment(self):
        T = TRAINING_WINDOW_SIZE
        for currentDay, df in enumerate(Test.logReturns, start=750):
            # this should match exactly what updateLogReturns builds
            expected = pd.RangeIndex(
                start=currentDay - T + 1,
                stop=currentDay + 1
            )
            with self.subTest(day=currentDay):
                # 1) quick boolean check:
                self.assertTrue(
                    df.index.equals(expected),
                    msg=f"Index mismatch on day {currentDay}: got {df.index}, expected {expected}"
                )
                # 2) or for a richer diff:
                assert_index_equal(df.index, expected)

    def testLogReturnsAccuracy(self):
        prices = Test.prices
        T = TRAINING_WINDOW_SIZE

        for currentDay, logReturns in enumerate(Test.logReturns, start = 750):
            pricesInWindow = prices[:, currentDay - T : currentDay + 1]
            actualLogReturnsInWindowNp = np.log(pricesInWindow[:, 1:] / pricesInWindow[:, :-1])

            index = pd.RangeIndex(start=currentDay - T + 1, stop=currentDay + 1)

            actualLogReturnsInWindow = pd.DataFrame(actualLogReturnsInWindowNp.T,
                                      index=index,
                                      columns=[f"inst_{i}" for i in range(actualLogReturnsInWindowNp.shape[0])])

            assert_frame_equal(logReturns, actualLogReturnsInWindow,
                               check_dtype=True,
                               check_exact=True,
                               check_column_type=True,
                               check_names=True
                               )

    def testDailyHistoryMatchExpected(self):
        self.assertEqual(
            len(self.gmGreeksHistory),
            len(self.actualGreeksHistory),
            "Mismatch in number of days tested, fix the test class"
        )

        for day_idx, (gm_dict, actual_dict) in enumerate(
                zip(self.gmGreeksHistory, self.actualGreeksHistory),
                start=750
        ):
            self.assertEqual(
                set(gm_dict.keys()),
                set(actual_dict.keys()),
                msg=f"Instrument keys differ on day {day_idx}"
            )

            for inst_key in gm_dict:
                gm_df = gm_dict[inst_key]
                actual_df = actual_dict[inst_key]

                assert_frame_equal(gm_df, actual_df,
                                   check_dtype=True,
                                   check_exact=True,
                                   check_column_type=True,
                                   check_names=True
                                   )

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

    @staticmethod
    def produceDayGetGreeksDictLikeGM(prices, index):
        greeksDict = Test.produceDayGreeksDict(prices)

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

    @staticmethod
    def getDailyGreeksFromMemberVar(greeksManager) -> dict[str, np.ndarray]:
        return {
            greekName: greek.getGreeks()
            for greekName, greek in greeksManager.greeks.items()
        }

    @staticmethod
    def produceGreeksHistory(prices, index):
        greeksPerDay = []
        for dayIndex in range(prices.shape[1] - TRAINING_WINDOW_SIZE - 1, prices.shape[1]):
            greeksPerDay.append(Test.produceDayGreeksDict(prices[:, :dayIndex + 1]))

        greek_names = list(greeksPerDay[0].keys())

        greeksPerDayNp = np.array([
            np.swapaxes([day_dict[g] for g in greek_names], 0, 1)
            for day_dict in greeksPerDay
        ])

        exogDict = {}
        for inst in range(prices.shape[0]):
            exogDict[f"inst_{inst}"] = pd.DataFrame(
                greeksPerDayNp[:-1, inst, :],
                columns=greek_names,
                index=index
            )

        return exogDict