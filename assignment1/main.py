# %%
import os
import sys

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

exp_num = 3
base_dir = "assignment1"
model_filename = f'{base_dir}/model_{exp_num}.pkl'
imputer_filename = f'{base_dir}/imputer_{exp_num}.pkl'
scaler_filename = f'{base_dir}/scaler_{exp_num}.pkl'
encoder_filename = f'{base_dir}/encoder_{exp_num}.pkl'

if __name__ == '__main__':
    pass

# %%

def preprocess_data(data, imputer=None, scaler=None, encoders=None):
    # -------------------------- 请调整你的数据预处理过程 ---------------------------
    ## 输入：
    #### data 为 pandas.DataFrame类型数据
    #### imputer 为缺失值填充方式
    #### scaler 为数据归一化方式
    ## 输出：
    #### data_norm 为处理后的数据，为 pandas.DataFrame类型数据

    column_name = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
                   'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
                   'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
                   ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
                   'Schooling']
    # data = data.drop(["Country", "Status"], axis=1)

    if imputer is None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = imputer.fit(data[column_name])
    data[column_name] = imputer.transform(data[column_name])

    if scaler is None:
        scaler = MinMaxScaler()
        scaler = scaler.fit(data[column_name])
    # data_norm = pd.DataFrame(scaler.transform(data[column_name]), columns=data.columns)
    data[column_name] = scaler.transform(data[column_name])

    # 将 country, status 变成 one-hot
    categoricals = ["Country", "Status"]
    if encoders is None:
        encoders = []
        for label in categoricals:
            le = LabelEncoder()
            new_labels = le.fit_transform(data[label]).reshape(-1, 1)

            ohe = OneHotEncoder(categories='auto')
            feature_arr = ohe.fit_transform(new_labels).toarray()
            feature_labels = [f"{label}_{cls_label}" for cls_label in le.classes_]
            features = pd.DataFrame(feature_arr, columns=feature_labels)

            data = data.drop(label, axis=1)
            data = pd.concat([data, features], axis=1)

            encoders.append([le, ohe])
    else:
        for idx, label in enumerate(categoricals):
            le, ohe = encoders[idx]
            new_labels = le.transform(data[label]).reshape(-1, 1)
            feature_arr = ohe.transform(new_labels).toarray()
            feature_labels = [f"{label}_{cls_label}" for cls_label in le.classes_]
            features = pd.DataFrame(feature_arr, columns=feature_labels)
            data = data.drop(label, axis=1)
            data = pd.concat([data, features], axis=1)

    data = data.drop(['Year'], axis=1)
    data_norm = data

    return data_norm, imputer, scaler, encoders


def predict(test_data):
    # -------------------------- 请加载您最满意的模型 ---------------------------
    # 加载模型(请加载你认为的最佳模型)
    # 加载模型,加载请注意 filename 是相对路径, 与当前文件同级。
    # test_data 为 pandas.DataFrame类型数据
    loaded_model = joblib.load(model_filename)
    imputer = joblib.load(imputer_filename)
    scaler = joblib.load(scaler_filename)

    test_data_norm = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values
    predictions = loaded_model.predict(test_x)

    return predictions

# %%

def model_fit(train_data):
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler, encoders = preprocess_data(train_data)

    train_x = train_data_norm.values

    # regressor = LinearRegression()
    # regressor.fit(train_x, train_y)

    regressor = GradientBoostingRegressor()
    regressor.fit(train_x, train_y)

    joblib.dump(regressor, model_filename)
    joblib.dump(imputer, imputer_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(encoders, encoder_filename)

    return regressor


# %%

def predict(test_data, filename):
    loaded_model = joblib.load(model_filename)
    imputer = joblib.load(imputer_filename)
    scaler = joblib.load(scaler_filename)
    encoders = joblib.load(encoder_filename)

    test_data_norm, _, _, _ = preprocess_data(test_data, imputer, scaler, encoders)
    test_x = test_data_norm.values
    predictions = loaded_model.predict(test_x)

    return predictions


# %%
# train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
# model = model_fit(train_data)

### 2.4 模型性能评估

# %%
def test(test_data):
    label = test_data.loc[:, 'Adult Mortality']
    data = test_data.iloc[:, :-1]
    y_pred = predict(data, './model.pkl')
    r2 = r2_score(label, y_pred)
    mse = mean_squared_error(label, y_pred)
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))

# test()

# %%
def validate(train_data):
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler, encoders = preprocess_data(train_data)
    train_x = train_data_norm.values

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=37)

    regressor = GradientBoostingRegressor(
        learning_rate=0.01,
        n_estimators=700,
        min_samples_leaf=35,
        min_samples_split=90,
        max_depth=5,
        max_features=180,
        subsample=1.0,
        random_state=42
    )
    regressor.fit(train_x, train_y)

    y_pred = regressor.predict(test_x)
    r2 = r2_score(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))
    print(regressor.feature_importances_)

train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
validate(train_data)

# %%
def main():
    train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
    model = model_fit(train_data)
    test(train_data)

main()

# %%
"""
grid search for GBR
"""
from sklearn import model_selection as cross_validation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv(f'{base_dir}/data/train_data.csv')
train_y = train_data.iloc[:, -1].values
train_data = train_data.drop(["Adult Mortality"], axis=1)
train_data_norm, imputer, scaler, encoders = preprocess_data(train_data)
train_x = train_data_norm.values

# %%
param_test1 = {'n_estimators': range(20, 101, 5)}
# gsearch1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1, min_samples_split=2,
#                         min_samples_leaf=1, max_depth=3, max_features=None, subsample=1.0, random_state=None),
#                         param_grid=param_test1, scoring='r2', cv=5)
gsearch1 = GridSearchCV(estimator=GradientBoostingRegressor(),
                        param_grid=param_test1, scoring='r2', cv=5, verbose=1)
gsearch1.fit(train_x, train_y)
print("results:\n", gsearch1.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch1.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch1.best_score_, "\n" + "-"*40)

# best_params_:
#  {'n_estimators': 35}
# ----------------------------------------
# best_score_:
#  0.50510200623396
# ----------------------------------------

# %%
param_test2 = {'max_depth': range(2, 14, 1), 'min_samples_split': range(2, 128, 8)}
gsearch2 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1, n_estimators=35, min_samples_leaf=1,
                        max_features=None, subsample=1.0, random_state=None),
                        param_grid=param_test2, scoring='r2', cv=5, verbose=1)
gsearch2.fit(train_x, train_y)
print("results:\n", gsearch2.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch2.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch2.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'max_depth': 5, 'min_samples_split': 2}
# ----------------------------------------
# best_score_:
#  0.5212259832620839

# %% 尝试使用 pipeline
# from sklearn.pipeline import Pipeline
param_test2 = {"GBR__max_depth": range(2, 14, 1), "GBR__min_samples_split": range(2, 128, 4)}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(learning_rate=0.1, n_estimators=35, min_samples_leaf=1,
                        max_features=None, subsample=1.0, random_state=None))])
gsearch2 = GridSearchCV(pipeline, param_test2, scoring='r2', n_jobs=40, verbose=1)
gsearch2.fit(train_x, train_y)
print("results:\n", gsearch2.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch2.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch2.best_score_, "\n" + "-"*40)

# 并行了之后快了不止一点半点！
# best_params_:
#  {'GBR__max_depth': 5, 'GBR__min_samples_split': 2}
# ----------------------------------------
# best_score_:
#  0.52202111112728

# %%
param_test3 = {"GBR__min_samples_split": range(2, 128, 8), "GBR__min_samples_leaf": range(0, 101, 5)}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(learning_rate=0.1, n_estimators=35, max_depth=5,
                                                       max_features=None, subsample=0.8, random_state=None))])

gsearch3 = GridSearchCV(pipeline, param_test3, scoring='r2', n_jobs=40, verbose=1)
gsearch3.fit(train_x, train_y)
print("results:\n", gsearch3.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch3.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch3.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__min_samples_leaf': 35, 'GBR__min_samples_split': 90}
# ----------------------------------------
# best_score_:
#  0.5458911529872628

# %%
param_test4 = {"GBR__max_features": range(1, 220, 2)}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(learning_rate=0.1, n_estimators=35, max_depth=5, min_samples_leaf=35,
                                                       subsample=0.8, random_state=None, min_samples_split=90))])

gsearch4 = GridSearchCV(pipeline, param_test4, scoring='r2', n_jobs=40, verbose=1)
gsearch4.fit(train_x, train_y)
print("results:\n", gsearch4.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch4.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch4.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__max_features': 179}
# ----------------------------------------
# best_score_:
#  0.5464334204640455

# %%
param_test5 = {"GBR__subsample": [0.6,0.7,0.75,0.8,0.85,0.9]}
pipeline = Pipeline([('GBR', GradientBoostingRegressor(learning_rate=0.1, n_estimators=35, max_depth=5, min_samples_leaf=35,
                                                       max_features=180, random_state=None, min_samples_split=90))])

gsearch5 = GridSearchCV(pipeline, param_test5, scoring='r2', n_jobs=40, verbose=1)
gsearch5.fit(train_x, train_y)
print("results:\n", gsearch5.cv_results_, "\n" + "-"*40)
print("best_params_:\n", gsearch5.best_params_, "\n" + "-"*40)
print("best_score_:\n", gsearch5.best_score_, "\n" + "-"*40)

# ----------------------------------------
# best_params_:
#  {'GBR__subsample': 0.8}
# ----------------------------------------
# best_score_:
#  0.5396721399798385

# %%
"""
experimental
"""

sys.exit()

# %% lab
data = pd.read_csv(f'{base_dir}/data/train_data.csv')
data, _, _, _ = preprocess_data(data)

# %%

