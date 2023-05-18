import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값 조정

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

print(x, y) # 
print('y의 라벨 값 : ', np.unique(y))   # y의 라벨 값 :  [0 1 2]

#2. 모델구성
model = SVC()
#model = LinearSVC()

#3. 컴파일, 훈련
model.fit(x, y)


#4. 평가, 예측
result = model.score(x, y)
#print('결과 acc : ', result)
print('결과 r2 : ', result)

## SVC result ##
# 결과 r2 :  0.7078651685393258

## Linear SVC result ##
# 결과 acc :  0.7696629213483146

