import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값 조정

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100, shuffle=True
)

## scaler 적용
scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = SVR()   # 회귀분석
#model = LinearSVR()

#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)
print('r2 : ', result)


## SVR result ##
# y의 라벨 값 :  [0.14999 0.175   0.225   ... 4.991   5.      5.00001]
# 결과 acc :  -0.01658668690926901

## LinearSVR result ##
# 결과 acc :  -4.008692834346897

## tf result ##
# loss :  0.6126309633255005
# r2 스코어 :  0.5453226457315958
# 걸린 시간 :  49.679261684417725

## MinMaxScaler 적용 : 0.6797978499327231
## StandardScaler 적용 : 0.7525300692785053
## MaxAbsScaler 적용 : 0.6008457676997941
## RobustScaler 적용 : 0.6886468428550829
