import numpy as np
from sklearn.svm import LinearSVC, LinearSVR    # C=class / R=
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 1]


#2. 모델구성
model = Perceptron()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가예측
result = model.score(x_data, y_data)
print('모델 scores : ', result)

y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)

acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)

### result ###
# 모델 scores :  1.0
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 1]
# acc :  1.0