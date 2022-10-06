# 第二次实验，使用 GBM，但是放弃国家和status数据

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
from sklearn.ensemble import GradientBoostingRegressor
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
        dropped = ["Country", 'Year']
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

    regressor = GradientBoostingRegressor(
        learning_rate=0.1/10/2,
        min_samples_leaf=10,
        max_features=17,
        n_estimators=40*2*2*2*2*2,
        subsample=0.8,
        random_state=42,
        max_depth=9,
        min_samples_split=50
    )
    regressor.fit(train_x, train_y)

    y_pred = regressor.predict(test_x)
    r2 = r2_score(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))
    if verbose == 2:
        print(regressor.feature_importances_)

# %%
train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
validate(train_data, verbose=2)

# %%
train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
columns = list(train_data.columns)
columns.remove("Country")
columns.remove("Status")
columns.remove("Adult Mortality")
for col in columns:
    train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
    dropped = ["Country", "Status", col]
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler = preprocess_data(train_data, dropped=dropped)
    train_x = train_data_norm.values

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=37)

    regressor = GradientBoostingRegressor(
        learning_rate=0.1 / 10 / 2,
        min_samples_leaf=10,
        max_features=18,
        n_estimators=40 * 2 * 2 * 2 * 2 * 2,
        subsample=0.8,
        random_state=42,
        max_depth=9,
        min_samples_split=50
    )
    regressor.fit(train_x, train_y)

    y_pred = regressor.predict(test_x)
    r2 = r2_score(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    print("-"*40)
    print(f"Dropped {col}")
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))


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
param_test1 = {'GBR__n_estimators': range(1, 101, 1)}
# gsearch1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1, min_samples_split=2,
#                         min_samples_leaf=1, max_depth=3, max_features=None, subsample=1.0, random_state=None),
#                         param_grid=param_test1, scoring='r2', cv=5)
pipeline = Pipeline([('GBR', GradientBoostingRegressor(
                                        learning_rate=0.1,
                                        min_samples_leaf=10, max_features=None,
                                        subsample=0.8,
                                        random_state=42,
                                        max_depth=5,
                                        min_samples_split=20))])
gsearch1 = GridSearchCV(pipeline, param_test1, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch1.fit(train_x, train_y)
print("results:\n", gsearch1.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch1.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch1.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__n_estimators': 36}
# ----------------------------------------
# best_score_:
#  0.5238368253106038

# %%
param_test2 = {"GBR__max_depth": range(2, 20, 1), "GBR__min_samples_split": range(2, 256, 8)}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(
                                            learning_rate=0.1,
                                            min_samples_leaf=10,
                                            max_features=None,
                                            n_estimators=38,
                                            subsample=0.8,
                                            random_state=42,
                                            # max_depth=5,
                                            # min_samples_split=20
                                            ))])
gsearch2 = GridSearchCV(pipeline, param_test2, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch2.fit(train_x, train_y)
print("results:\n", gsearch2.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch2.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch2.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__max_depth': 9, 'GBR__min_samples_split': 50}
# ----------------------------------------
# best_score_:
#  0.5450846822274051

# %%
param_test3 = {"GBR__min_samples_split": range(2, 128, 8), "GBR__min_samples_leaf": range(1, 101, 5)}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(
                                            learning_rate=0.1,
                                            # min_samples_leaf=10,
                                            max_features=None,
                                            n_estimators=38,
                                            subsample=0.8,
                                            random_state=42,
                                            max_depth=9,
                                            # min_samples_split=50
                                            ))])
gsearch3 = GridSearchCV(pipeline, param_test3, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch3.fit(train_x, train_y)
print("results:\n", gsearch3.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch3.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch3.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__min_samples_leaf': 10, 'GBR__min_samples_split': 50}
# ----------------------------------------
# best_score_:
#  0.5450846822274051

# %%
param_test4 = {"GBR__max_features": range(1, 19, 1)}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(
                                            learning_rate=0.1,
                                            min_samples_leaf=10,
                                            # max_features=None,
                                            n_estimators=38,
                                            subsample=0.8,
                                            random_state=42,
                                            max_depth=9,
                                            min_samples_split=50
                                            ))])
gsearch4 = GridSearchCV(pipeline, param_test4, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch4.fit(train_x, train_y)
print("results:\n", gsearch4.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch4.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch4.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__max_features': 18}
# ----------------------------------------
# best_score_:
#  0.5450846822274051

# %%
param_test5 = {"GBR__subsample": [0.6,0.7,0.75,0.8,0.85,0.9]}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(
                                            learning_rate=0.1,
                                            min_samples_leaf=10,
                                            max_features=18,
                                            n_estimators=38,
                                            # subsample=0.8,
                                            random_state=42,
                                            max_depth=9,
                                            min_samples_split=50
                                            ))])
gsearch5 = GridSearchCV(pipeline, param_test5, scoring='r2', cv=5, n_jobs=40, verbose=1)
gsearch5.fit(train_x, train_y)
print("results:\n", gsearch5.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch5.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch5.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__subsample': 0.8}
# ----------------------------------------
# best_score_:
#  0.5450846822274051

# %%
# check data
train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')

# %%
import matplotlib.pyplot as plt

for col in train_data.columns:
    if col in ["Country", "Status"]:
        continue
    plt.boxplot(train_data[col].values)
    plt.title(col)
    plt.show()

