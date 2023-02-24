import unittest

from housing_price import score


def fun(x):
    return x + 1


class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(fun(3), 4)

    def test_ingest_data(self):
        flag_var, rmse, mae, r2 = score.eval_metrics(
            ".././housing_price/datasets/housing/",
            "../../artifacts/",
            "../../artifacts/",
        )
        self.assertTrue(
            flag_var, "Function was not properly executed! Please check the code."
        )
        self.assertIsNotNone(rmse, "RMSE should be not none!")
        self.assertIsNotNone(mae, "MAE should not be none!")
        self.assertIsNotNone(r2, "R2 should not be none!")


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
