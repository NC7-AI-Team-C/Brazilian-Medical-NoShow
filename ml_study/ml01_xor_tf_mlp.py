import numpy as np
from sklearn.svm import LinearSVC, LinearSVR    # C=class / R= Regression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]



#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=2))  # MLP(Multi Layer Perceptron)
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가예측
loss, acc = model.evaluate(x_data, y_data)
y_predict = model.predict(x_data)
print('loss : ', loss)
print(x_data, '의 예측결과 : ', y_predict)
print('acc : ', acc)

### result ###
# loss :  0.13114160299301147
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.29182446]
#  [0.9502114 ]
#  [0.93200547]
#  [0.05636489]]
# acc :  1.0
