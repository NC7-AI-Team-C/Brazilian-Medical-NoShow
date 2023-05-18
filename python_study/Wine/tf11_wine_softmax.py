import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time

#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
# 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) # (124, 13) (124,)
print(x_test.shape, y_test.shape)   # (54, 13) (54,)
print(y_test)
print(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)






