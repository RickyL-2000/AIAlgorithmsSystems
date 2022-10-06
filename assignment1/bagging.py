# 使用 bagging 模型
# 参考 https://blog.csdn.net/qq_38299170/article/details/103833113

# %%
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import joblib

exp_num = 5
base_dir = "assignment1"
model_filename = f'{base_dir}/model_{exp_num}.pkl'
imputer_filename = f'{base_dir}/imputer_{exp_num}.pkl'
scaler_filename = f'{base_dir}/scaler_{exp_num}.pkl'
encoder_filename = f'{base_dir}/encoder_{exp_num}.pkl'

if __name__ == '__main__':
    pass

# %%
def preprocess_data(data, imputer=None, scaler=None, dropped=None):
    column_name = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
                   'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
                   'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
                   ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
                   'Schooling']
    if dropped is None:
        dropped = ["Country", "Year"]
    data = data.drop(dropped, axis=1)
    # data = data.drop(["Country"], axis=1)
    for col in dropped:
        if col in column_name:
            column_name.remove(col)

    if imputer is None:
        # imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        imputer = imputer.fit(data[column_name])
    data[column_name] = imputer.transform(data[column_name])

    # le = LabelEncoder()
    # new_labels = le.fit_transform(data["Status"]).reshape(-1, 1)
    # ohe = OneHotEncoder(categories='auto')
    # feature_arr = ohe.fit_transform(new_labels).toarray()
    # feature_labels = [f"Status_{cls_label}" for cls_label in le.classes_]
    # features = pd.DataFrame(feature_arr, columns=feature_labels)
    # data = data.drop("Status", axis=1)
    # data = pd.concat([data, features], axis=1)
    status2oh = {'Developing': 0, 'Developed': 1}
    data['Status'] = data['Status'].map(status2oh)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)

    # data_norm = data_norm.drop(['Year'], axis=1)

    return data_norm, imputer, scaler

# %%
def validate(train_data, verbose=0):
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler= preprocess_data(train_data)
    train_x = train_data_norm.values

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=37)

    regressor = BaggingRegressor(
        base_estimator=None,
        n_estimators=480,
        max_samples=0.5,
        max_features=0.6,
        bootstrap=False,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        random_state=42
    )
    regressor.fit(train_x, train_y)

    y_pred = regressor.predict(test_x)
    r2 = r2_score(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))

train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
validate(train_data)

# %%
"""
调参
"""
from sklearn import model_selection as cross_validation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
train_y = train_data.iloc[:, -1].values
train_data = train_data.drop(["Adult Mortality"], axis=1)
train_data_norm, imputer, scaler = preprocess_data(train_data)
train_x = train_data_norm.values

# %%
param_test1 = {'BR__n_estimators': range(1, 101, 1)}
pipeline = Pipeline([('BR', BaggingRegressor(
    base_estimator=None,
    n_estimators=10,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=False,
    warm_start=False,
    random_state=42
))])
gsearch1 = GridSearchCV(pipeline, param_test1, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch1.fit(train_x, train_y)
print("results:\n", gsearch1.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch1.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch1.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__n_estimators': 33}
# ----------------------------------------
# best_score_:
#  0.5124851837973659
# ----------------------------------------

# %%
param_test2 = {'BR__max_samples': np.arange(0.5, 1.05, 0.05), 'BR__max_features': np.arange(0.3, 1.05, 0.05)}
pipeline = Pipeline([('BR', BaggingRegressor(
    base_estimator=None,
    n_estimators=33,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=False,
    warm_start=False,
    random_state=42
))])
gsearch2 = GridSearchCV(pipeline, param_test2, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch2.fit(train_x, train_y)
print("results:\n", gsearch2.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch2.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch2.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'BR__max_features': 0.5, 'BR__max_samples': 0.8000000000000003}
# ----------------------------------------
# best_score_:
#  0.5435421846482762
# ----------------------------------------

# %%
param_test3 = {'BR__bootstrap': [True, False],
               'BR__bootstrap_features': [True, False],
               'BR__oob_score': [True, False],
               'BR__warm_start': [True, False], }
pipeline = Pipeline([('BR', BaggingRegressor(
    base_estimator=None,
    n_estimators=33,
    max_samples=0.8,
    max_features=0.5,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,
    warm_start=False,
    random_state=42
))])
gsearch3 = GridSearchCV(pipeline, param_test3, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch3.fit(train_x, train_y)
print("results:\n", gsearch3.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch3.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch3.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'BR__bootstrap': True, 'BR__bootstrap_features': False, 'BR__oob_score': True, 'BR__warm_start': False}
# ----------------------------------------
# best_score_:
#  0.5435421846482762
# ----------------------------------------

# %%
param_test4 = {'BR__max_samples': np.arange(0.5, 1.0, 0.05),
               'BR__max_features': np.arange(0.5, 1.0, 0.05),
               # 'BR__bootstrap': [True, False],
               'BR__bootstrap_features': [True, False],
               'BR__oob_score': [True, False],
               # 'BR__warm_start': [True, False],
               }
pipeline = Pipeline([('BR', BaggingRegressor(
    base_estimator=None,
    n_estimators=480,
    max_samples=0.8,
    max_features=0.5,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,
    warm_start=False,
    random_state=42
))])
gsearch4 = GridSearchCV(pipeline, param_test4, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch4.fit(train_x, train_y)
print("results:\n", gsearch4.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch4.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch4.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'BR__bootstrap': False, 'BR__bootstrap_features': False, 'BR__max_features': 0.6000000000000001,
#  'BR__max_samples': 0.5, 'BR__oob_score': False}
# ----------------------------------------
# best_score_:
#  0.5426998577494098
# ----------------------------------------
