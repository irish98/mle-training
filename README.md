# mle-training
Assignment 3.1

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
This code has python files refactored as scripts and are able to install as a package.
All the python scripts are available at `src/housing_price`.
This folder contains `datasets` which has the required datsets along with train and test datasets as csv files.

`housing_price` folder is refactored to a package so it contains `__init__.py`.

This package also contains `tests` folder which has some functional and unit tests for this project.

For installing the `housing_price` package user can just run the below command:

```
(mle-dev) (root-folder)$python setup.py install
```
`setup.py` contains code which will install the required packages and can use as a library.

For running scripts user can run the below command:
I have used argparse for taking the arguments from the command line:

```
(mle-dev) (script-folder)$python3 <script>.py --args
```

User can run the below code to see what are the options each script takes:

```
(mle-dev) (script-folder)$python <script>.py --help
or
(mle-dev) (script-folder)$python <script>.py -h
```

For running the tests user can just run the below command:

```
(mle-dev) (tests-folder)$py.test
```

These are the required commands for executing this project.

