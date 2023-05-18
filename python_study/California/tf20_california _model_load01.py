import numpy as np
from keras.models import Sequential, load_model # load_model 불러오기
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
# from sklearn.datasets import load_boston  # 윤리적인 문제로 제공 안됌.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # 오류시 인증서 무시 명령어
import time


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
    x, y, test_size=0.2, random_state=100, shuffle=True
)
print(x_train.shape)    #(14447, 8)
print(y_train.shape)    #(14447,)

#2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=8))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(1))

# model.save('./_save/tf17_california.h5')    # h5로 모델 저장
   # save 했던 경로및 파일을 똑같이 불러오기

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# ## earlystopping
# from keras.callbacks import EarlyStopping
# earlystopping = EarlyStopping(monitor='val_loss', patience=100, 
#                               mode='min', verbose=1, restore_best_weights=True) # mode의 default = auto // restore_best_weight의 default는 False 이므로 True로 꼭꼭꼭!!!변경!!

# start_time = time.time()
# hist=model.fit(x_train, y_train, epochs=1000, batch_size=200,    # hist
#           validation_split=0.2,
#           callbacks=[earlystopping],
#           verbose=1) # validation data => 0.2 (train = 0.6/ test = 0.2)

# end_time = time.time() - start_time

model = load_model('./_save/tf20_california.h5')    # model

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2_score)
# print('걸린 시간 : ', end_time) # load 할 시 time은 작동되지 않음 & model만 저장되기 때문에

## epochs 1000일 때 & patience 100일 때
# Epoch 981: early stopping
# loss :  0.5784325003623962
# r2 스코어 :  0.5707038331945751
# 걸린 시간 :  87.71940279006958

# Epoch 403: early stopping
# loss :  0.5962461829185486
# r2 스코어 :  0.557482965434179
# 걸린 시간 :  44.80277347564697

## load_model로 해서 running 함
# Epoch 225: early stopping
# r2 스코어 :  0.5432691320580564
# 걸린 시간 :  25.024884939193726
# loss :  0.6153980493545532

