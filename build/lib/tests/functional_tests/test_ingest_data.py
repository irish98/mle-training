import unittest
from os.path import exists

from housing_price import ingest_data


def func(x):
    return x + 1


class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(func(3), 4)

    def test_ingest_data(self):
        self.assertTrue(
            ingest_data.prepare_train(".././housing_price/datasets/housing/"),
            "Function was not properly executed! Please check the code.",
        )
        train_exists = exists(".././housing_price/datasets/housing/train/train.csv")
        self.assertTrue(train_exists, "File not exists! Please check the code!")
        test_exists = exists(".././housing_price/datasets/housing/test/test.csv")
        self.assertTrue(test_exists, "File not exists! Please check the code!")


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
