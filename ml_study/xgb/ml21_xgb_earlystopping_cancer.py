import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=64
)

### kfold ###
n_splits = 21
random_state = 64
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='error')

#4. 출력(평가, 예측)
result = model.score(x_test, y_test)
print('acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv acc : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

# acc :  0.956140350877193

# cv acc :  [1.         1.         0.95454545 1.         0.90909091 0.95454545
#  0.95454545 1.         0.95454545 1.         0.86363636 0.90909091
#  1.         1.         1.         1.         1.         1.
#  1.         0.95238095 0.9047619 ]
# cv pred :  [1 1 1 1 1 1 1 0 0 0 0 1 1 0 1 0 1 1 1 1 0 1 0 1 0 0 0 0 1 1 0 1 0 1 1 0 0
#  0 1 1 1 1 0 1 0 1 0 1 1 1 1 0 0 1 1 1 1 1 1 0 1 0 0 0 1 1 1 1 1 1 0 1 1 1
#  0 1 1 1 0 1 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1
#  0 0 1]
# cv pred acc :  0.9210526315789473
