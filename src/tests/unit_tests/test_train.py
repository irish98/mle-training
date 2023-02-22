import unittest
from os.path import exists

from housing_price import train


def fun(x):
    return x + 1


class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(fun(3), 4)

    def test_ingest_data(self):
        self.assertTrue(
            train.training_data(
                ".././housing_price/datasets/housing/", "../../artifacts/"
            ),
            "Code not executed properly. Please check the code and rectify errors!",
        )
        model_exists = exists("../../artifacts/housing_model.pkl")
        self.assertTrue(model_exists, "File not exists! Please check the code!")


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
