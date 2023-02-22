import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_folder",
    default="artifacts",
    help="input path where the model pickle files are stored",
)

parser.add_argument(
    "--datasetpath",
    default=os.path.join("data", "train_test"),
    help="path for the folder where the train and test datas are present",
)

parser.add_argument(
    "--noconsolelog",
    help="not to write log files to console",
    action="store_true",
)
if parser.parse_args().noconsolelog:
    logging.basicConfig(
        filename="logs/logs.log",
        level=logging.DEBUG,
        format="%(pathname)s:%(levelname)s:%(message)s",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(pathname)s:%(levelname)s:%(message)s",
    )

model_file_path = parser.parse_args().model_folder
dataset_path = parser.parse_args().datasetpath

model_pickle = open(os.path.join(model_file_path, "final_model.pkl"), "rb")
final_model = pickle.load(model_pickle)

imputer_pickle = open(os.path.join(model_file_path, "imputer.pkl"), "rb")
imputer = pickle.load(imputer_pickle)

strat_test_set = pd.read_csv(os.path.join(dataset_path, "test.csv"))
# print(strat_test_set.info())
# print(strat_test_set.head())

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_num.drop("Unnamed: 0",axis=1, inplace=True)
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
X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True)
)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
logging.debug(f"model rmse score is {final_rmse}")
