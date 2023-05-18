#1. 데이터(numpy로 x,y 값을 array해서 입력)
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1))   # 입력층 / dim은 차원
model.add(Dense(50))                # 히든레이어
model.add(Dense(25))                # model.add(Dense(25))
model.add(Dense(15))                # model.add(Dense(15)) 
model.add(Dense(1))                 # 출력층

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)        # epochs=1000


#4. 평가, 예측
loss = model.evaluate(x, y)         # 실제값, 예측값 계산해봐바
print('loss : ', loss)              # loss :  0.0

result = model.predict([4])
print('4의 예측값 : ', result)       # 4의 예측값 :  [[4.]]
