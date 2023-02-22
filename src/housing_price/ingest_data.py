import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib  # pyright: ignore
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This function downloads housing file, extracts the housing.csv file and
    saves it in housing_path.

    :param housing_url : URL of the housing dataet
    :param housing_path : Directory to save the dataset
    """

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    loads the housing data from housing_path and returns dataframe object with
    housing data

    :param housing_path: path to the dataset
    :return : pandas dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=os.path.join("data", "train_test"),
        help="Enter the output path for train and test datasets",
    )

    parser.add_argument(
        "--noconsolelog",
        help="not to write log files to console",
        action="store_true",
    )
    if parser.parse_args().noconsolelog:
        logging.basicConfig(
            filename="logs/logfile.log",
            level=logging.DEBUG,
            format="%(pathname)s:%(levelname)s:%(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(pathname)s:%(levelname)s:%(message)s",
        )

    path = parser.parse_args().path

    fetch_housing_data()
    housing = load_housing_data()

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    os.makedirs(path, exist_ok=True)
    strat_train_set.to_csv(os.path.join(path, "train.csv"))
    strat_test_set.to_csv(os.path.join(path, "test.csv"))
    logging.debug("Train test data prepared and stored")
