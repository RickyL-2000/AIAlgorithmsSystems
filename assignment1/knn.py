# 使用 knn 模型

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
from sklearn.neighbors import KNeighborsRegressor
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

    le = LabelEncoder()
    new_labels = le.fit_transform(data["Status"]).reshape(-1, 1)
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(new_labels).toarray()
    feature_labels = [f"Status_{cls_label}" for cls_label in le.classes_]
    features = pd.DataFrame(feature_arr, columns=feature_labels)
    data = data.drop("Status", axis=1)
    data = pd.concat([data, features], axis=1)

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

    regressor = KNeighborsRegressor(
        n_neighbors=18,
        weights='distance',
        algorithm='kd_tree',
        leaf_size=5,
        metric='manhattan'
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
param_test1 = {'KNN__n_neighbors': range(3, 15, 1)}
pipeline = Pipeline([('KNN', KNeighborsRegressor(
    # n_neighbors=5,
    weights='distance',
    algorithm='kd_tree',
    leaf_size=30,
    metric='euclidean'
))])
gsearch1 = GridSearchCV(pipeline, param_test1, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch1.fit(train_x, train_y)
print("results:\n", gsearch1.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch1.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch1.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'KNN__n_neighbors': 10}
# ----------------------------------------
# best_score_:
#  0.46504090954323685
# ----------------------------------------

# %%
param_test2 = {'KNN__weights': ["uniform", "distance"]}
pipeline = Pipeline([('KNN', KNeighborsRegressor(
    n_neighbors=10,
    # weights='distance',
    algorithm='kd_tree',
    leaf_size=30,
    metric='euclidean'
))])
gsearch2 = GridSearchCV(pipeline, param_test2, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch2.fit(train_x, train_y)
print("results:\n", gsearch2.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch2.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch2.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'KNN__weights': 'distance'}
# ----------------------------------------
# best_score_:
#  0.46504090954323685
# ----------------------------------------

# %%
param_test3 = {'KNN__leaf_size': range(1, 120, 1)}
pipeline = Pipeline([('KNN', KNeighborsRegressor(
    n_neighbors=10,
    weights='distance',
    algorithm='auto',
    # leaf_size=30,
    metric='euclidean'
))])
gsearch3 = GridSearchCV(pipeline, param_test3, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch3.fit(train_x, train_y)
print("results:\n", gsearch3.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch3.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch3.best_score_, "\n" + "-"*40)


# %%
param_test4 = {'KNN__metric': ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"]}
pipeline = Pipeline([('KNN', KNeighborsRegressor(
    n_neighbors=10,
    weights='distance',
    algorithm='kd_tree',
    leaf_size=30,
    # metric='euclidean'
))])
gsearch4 = GridSearchCV(pipeline, param_test4, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch4.fit(train_x, train_y)
print("results:\n", gsearch4.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch4.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch4.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'KNN__metric': 'manhattan'}
# ----------------------------------------
# best_score_:
#  0.500410228397995
# ----------------------------------------

# %%
param_test5 = {
    'KNN__n_neighbors': range(3, 20, 1),
    'KNN__weights': ["uniform", "distance"],
    'KNN__leaf_size': range(5, 60, 5),
    'KNN__algorithm': ['brute', 'kd_tree', 'ball_tree'],
    'KNN__metric': ["euclidean", "manhattan"]
}
pipeline = Pipeline([('KNN', KNeighborsRegressor(
    # n_neighbors=10,
    # weights='distance',
    algorithm='kd_tree',
    leaf_size=30,
    # metric='euclidean'
))])
gsearch5 = GridSearchCV(pipeline, param_test5, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch5.fit(train_x, train_y)
print("results:\n", gsearch5.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch5.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch5.best_score_, "\n" + "-"*40)
