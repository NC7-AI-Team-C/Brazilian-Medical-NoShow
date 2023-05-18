import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(77)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=64
)

### kfold ###
n_splits = 5
random_state = 64
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

param = [
    {'n_estimators' : [100, 200, 300], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12, 14, 16], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200, 300], 'min_samples_leaf' : [3, 5, 7, 10, 12]},
    {'min_samples_split' : [2, 3, 5, 10, 12], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200, 300],'n_jobs' : [-1, 2, 4]}
]

#2. 모델구성
from sklearn.model_selection import RandomizedSearchCV
rf_model = RandomForestRegressor()
# model = RandomizedSearchCV(rf_model, param, cv=kfold, verbose=1,
#                      refit=True, n_jobs=-1)
model = RandomForestRegressor(min_samples_split=3, n_jobs=4)

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

# print('최적의 파라미터 : ', model.best_params_)
# print('최적의 매개변수 : ', model.best_estimator_)
# print('베스트 스코어 : ', model.best_score_)
# print('모델 스코어 : ', model.score(x_test, y_test))
# print('걸린 시간 : ', end_time, '초')


# 최적의 파라미터 :  {'n_jobs': 4, 'min_samples_split': 3}
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3, n_jobs=4)
# 베스트 스코어 :  0.8005705388432667
# 모델 스코어 :  0.8105595194455746
# 걸린 시간 :  84.21564650535583 초

#4. 출력(평가, 예측)

result = model.score(x_test, y_test)
print('acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv acc : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('cv pred acc : ', r2)