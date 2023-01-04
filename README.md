# mle-training
Assignment 1.3

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

## Steps to setup the environment
- Create mle-dev environment using
```
  conda create --name 'mle-dev'
```
- Activate the environment using
```
  conda activate mle-dev
```
## Steps to format the code
- Use Black to format the code
```
  black nonstandardcode.py
```
- Use isort 
```
  isort nonstandardcode.py
```
- Use flake8
```
  flake8 nonstandardcode.py
```
