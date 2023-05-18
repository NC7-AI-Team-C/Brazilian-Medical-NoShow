import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# No. 8 ~ 9
# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 1, 2, 1, 1.1, 1.2, 1.4, 1.5, 1.6]])          # 대괄호 리스트를 대과호 리스트로 감싸다
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)  # (2, 10)   // 2행 10열
print(y.shape)  # (10,) // 10행 0열

x = x.transpose()
print(x.shape)  # (10, 2)   // 10행 2열

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))   # dim 은 변수가 몇개인가
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x, y, epochs=5000)

# 4. 평가, 예측
loss = model.evaluate(x, y) # loss :  7.275957722603643e-13
print('loss : ', loss)

result = model.predict([[10, 1.6]]) # [10]과 [1.6]의 예측값 :  [[20.000002]]
print('[10]과 [1.6]의 예측값 : ', result)



