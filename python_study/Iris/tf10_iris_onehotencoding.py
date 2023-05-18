import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data   # x = datasets['data'] 와 같은 명령어
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

### one hot encoding
y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
from keras.utils import to_categorical
print(x_train.shape, y_train.shape) # (105, 4) (105,)
print(x_test.shape, y_test.shape)   # (45, 4) (45,)
# print(y_test)
# print(x_test)

print(y)
print(y.shape)  # (150, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=4))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린 시간 : ', end_time) # 걸린 시간 :  2.7343313694000244

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)  # loss :  0.04617173224687576
print('acc : ', acc)    # acc :  0.9777777791023254



