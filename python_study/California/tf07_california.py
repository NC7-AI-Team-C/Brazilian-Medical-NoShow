import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
# from sklearn.datasets import load_boston  # 윤리적인 문제로 제공 안됌.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # 오류시 인증서 무시 명령어


#1. 데이터
# datasets = load_boston()
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
#'Population', 'AveOccup', 'Latitude', 'Longitude']

print(datasets.DESCR)

print(x.shape)  #(20640, 8)
print(y.shape)  #(20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape)    #(14447, 8)
print(y_train.shape)    #(14447,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
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


