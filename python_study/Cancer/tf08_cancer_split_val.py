import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)   # describe : 설명
print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
# ['평균 반경', '평균 질감', '평균 둘레', '평균 면적',
# '평균 매끄러움', '평균 조그만 정도', '평균 오목함',
# '평균 오목한 점의 수', '평균 대칭', '평균 프랙탈 차원',
# '반경 오차', '질감 오차', '둘레 오차', '면적 오차',
# '매끄러움 오차', '조그만 정도 오차', '오목함 오차',
# '오목한 점의 수 오차', '대칭 오차', '프랙탈 차원 오차',
# '최악 반경', '최악 질감', '최악 둘레', '최악 면적',
# '최악 매끄러움', '최악 조그만 정도', '최악 오목함',
# '최악 오목한 점의 수', '최악 대칭', '최악 프랙탈 차원']

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))   #이진분류는 무조건 아웃풋 레이어의
                                            #활성화 함수를 'sigmoid'로 해줘야 한다.


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics='mse')

start_time = time.time()
hist=model.fit(x_train, y_train, epochs=100, batch_size=200,
          verbose=0, validation_split=0.2)    # verbose로 결과값 추출 로딩 제어
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
# loss, acc, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# [실습] accurancy_score를 출력하라!
# y_predict를 반올롬 하기
y_predict = np.round(y_predict)   # y_predict를  round로 반올림
# y_predict = np.where(y_predict >= 0.5, 1, 0)    #y_predict where로 반올림

acc = accuracy_score(y_test, y_predict)

print('loss : ', loss)
print('Accuracy 스코어 : ', acc)
# print('mse : ', mse)
print('걸린 시간 : ', end_time)

# result
# loss :  [1.0612512826919556, 0.069186270236969]
# Accurancy 스코어 :  0.9298245614035088
# 걸린 시간 :  0.8653008937835693

# 시각화
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font = {'family': 'Segoe Print'}
plt.rc('font', **font)
plt.figure(figsize=(11, 9))
plt.plot(hist.history['loss'], marker='.', c='green', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='red', label='val_loss')
plt.title('loss & val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()






