import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성
model = Sequential()
model.add(Dense(7, input_dim=1))    # 입력층
model.add(Dense(100))               # 은닉층
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))                 #출력층

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') # mse= 평균제곱오차  // 최적의 파리미터 값을 찾아가는 알고리즘
model.fit(x, y, epochs=100)                 # w 값이 -(음수)가 나올때 'mae'를 하면 근사값이 나올 확률이 높아진다.(루트로 제곱을 없애줌)

# 4. 예측, 평가
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)