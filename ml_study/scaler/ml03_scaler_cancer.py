import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # 오류시 인증서 무시 명령어

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값 조정

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100, shuffle=True
)

## scaler 적용
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=200)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2_score)

## 그냥 돌렸을 때(scaler 적용 전) : -0.7965049516262908
## StandardScaler : 0.7618046474266981
## MinMaxScaler : 0.7635790918060448
## MaxAbsScaler : 0.7412225034102703
## RobustScaler : 0.7602003739214
