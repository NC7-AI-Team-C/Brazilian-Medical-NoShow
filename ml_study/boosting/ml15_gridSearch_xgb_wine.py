import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=64
)

### kfold ###
n_splits = 21
random_state = 64
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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
from xgboost import XGBRFClassifier
model = XGBRFClassifier()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# rf_model = RandomForestClassifier()
# model = GridSearchCV(rf_model, param, cv=kfold, verbose=1,
#                      refit=True, n_jobs=-1)


#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('최적의 파라미터 : ', model.best_params_)
print('최적의 매개변수 : ', model.best_estimator_)
print('베스트 스코어 : ', model.best_score_)
print('모델 스코어 : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')

#4. 출력(평가, 예측)
result = model.score(x_test, y_test)
print('acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold)   
print('cv acc : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

