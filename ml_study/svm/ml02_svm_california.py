import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값 조정

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (20640, 8) (20640,)
print('y의 라벨 값 : ', np.unique(y))  

#2. 모델구성
model = SVR()   # 회귀분석
#model = LinearSVR()

#3. 컴파일, 훈련
model.fit(x, y)


#4. 평가, 예측
result = model.score(x, y)
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