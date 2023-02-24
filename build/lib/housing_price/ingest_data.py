"""Importing required libraries."""
import argparse
import logging
import logging.config
import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import requests
# from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# import tarfile


# logger = logging.getLogger(__name__)

# Setting up a config
LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


def prepare_train(file_path):

    # Flag to verify the successful completion of code
    flag_var = False
    # DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = file_path
    # HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    """Given HOUSING_URL and housing_path, fetches housing_data file."""
    # os.makedirs(HOUSING_PATH, exist_ok=True)
    # tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    # req = requests.get(HOUSING_URL)
    # with open(tgz_path, "wb") as f:
    #     f.write(req.content)
    # urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    # print(tgz_path)
    # housing_tgz = tarfile.open(tgz_path)
    # housing_tgz.extractall(path=HOUSING_PATH)
    # housing_tgz.close()

    """Given housing_path, returns housing_data file."""
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    housing = pd.read_csv(csv_path)

    # fetch_housing_data()
    # housing = load_housing_data()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        """Given data, returns income cat proportions."""
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

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

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    strat_train_set.to_csv(file_path + "/train/train.csv")
    strat_test_set.to_csv(file_path + "/test/test.csv")

    print("Successfully created Training and Validation datasets!")
    flag_var = True

    return flag_var


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", help="give output folder/file path", required=True
    )
    parser.add_argument("--log-level", help="specify the log level")
    parser.add_argument("--log-path", help="use a log file or not")
    parser.add_argument(
        "--no-console-log",
        help="toggle whether or not to write logs to the console",
        action="store_true",
    )
    console_log = True
    log_level = "DEBUG"
    args = parser.parse_args()
    file_path = args.path
    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_file = args.log_path
    else:
        log_file = "../logs/ingest_data_logs.log"
    if args.no_console_log:
        console_log = False

    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(
        log_file=log_file, console=console_log, log_level=log_level
    )
    logger.info("Logging Test - Start")
    logger.info("Logging Test - Test 1 Done")
    logger.warning("Watch out!")

    prepare_train(file_path)

    logger.info(
        "Sucessfully completed downloading and preparing the training and validation data!"
    )
