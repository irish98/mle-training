"""Importing required libraries."""
import argparse
import logging
import logging.config

import joblib

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from six.moves import urllib
from sklearn import metrics
from sklearn.impute import SimpleImputer

# import os


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


def eval_metrics(data_path, output_path, model_path):
    """Evaluating the model"""

    # Flag to verify the successful completion of code
    flag_var = False

    # Loading data
    strat_train_set = pd.read_csv(data_path + "/train/train.csv")
    strat_test_set = pd.read_csv(data_path + "/test/test.csv")

    # drop labels for training set
    housing = strat_train_set.drop("median_house_value", axis=1)

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    imputer.transform(housing_num)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    # Loading model
    my_model_loaded = joblib.load(model_path + "/housing_model.pkl")
    final_predictions = my_model_loaded.predict(X_test_prepared)

    # Scoring the model
    rmse = np.sqrt(metrics.mean_squared_error(y_test, final_predictions))
    mae = metrics.mean_absolute_error(y_test, final_predictions)
    r2 = metrics.r2_score(y_test, final_predictions)
    print("Scoring the model --------")
    print(
        "RMSE: " + str(np.sqrt(metrics.mean_squared_error(y_test, final_predictions)))
    )
    print("MAE: " + str(metrics.mean_absolute_error(y_test, final_predictions)))
    print("R2 SCORE: " + str(metrics.r2_score(y_test, final_predictions)))
    print()

    flag_var = True

    return flag_var, rmse, mae, r2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="give model path", required=True)
    parser.add_argument(
        "-d", "--data_path", help="give data folder path", required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="give output folder path",
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
    model_path = args.model_path
    data_path = args.data_path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = "../../artifacts/"
    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_file = args.log_path
    else:
        log_file = "../logs/score_logs.log"
    if args.no_console_log:
        console_log = False

    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(
        log_file=log_file, console=console_log, log_level=log_level
    )
    logger.info("Logging Test - Start")
    logger.info("Logging Test - Test 1 Done")
    logger.warning("Watch out!")

    eval_metrics(data_path, output_path, model_path)
