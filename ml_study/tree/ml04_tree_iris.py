import numpy as np
from keras.models import Sequential
from sklearn.svm import SVR, LinearSVR
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from sklearn.datasets import load_boston  # 윤리적인 문제로 제공 안됌.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # 오류시 인증서 무시 명령어

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값 조정


#1. 데이터
datasets = load_iris()
x = datasets.data   # x = datasets['data'] 와 같은 명령어
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

## scaler 적용
#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=4))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=100,
               verbose=0, validation_split=0.2)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)



## 그냥 돌렸을 때(scaler 적용 전) : 1.0
## StandardScaler : 1.0
## MinMaxScaler : 1.0
## MaxAbsScaler : 1.0
## RobustScaler : 1.0

