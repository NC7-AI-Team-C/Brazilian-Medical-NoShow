import numpy as np
from sklearn.svm import LinearSVC, LinearSVR    # C=class / R=
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))  # sklearn의 Perceptron() 과 동일

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가예측
loss, acc = model.evaluate(x_data, y_data)
print('loss : ', loss)

y_predict = model.predict(x_data)
print(x_data, '의 예측결과 : ', y_predict)

print('acc : ', acc)

### result ###
# loss :  0.7689285278320312
# 1/1 [==============================] - 0s 65ms/step
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.4713217 ]
#  [0.5570648 ]
#  [0.71706754]
#  [0.78143686]]
# acc :  0.75
