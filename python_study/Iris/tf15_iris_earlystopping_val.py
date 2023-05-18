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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) # (105, 4) (105,)
print(x_test.shape, y_test.shape)   # (45, 4) (45,)
# print(y_test)
# print(x_test)

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
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
## earlystopping
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=100,
                              mode='min', verbose=1, restore_best_weights=True)

start_time = time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=100,
               verbose=1, validation_split=0.2,
               callbacks=[earlystopping])
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린 시간 : ', end_time)

# result
# loss :  0.04617173224687576
# acc :  0.9777777791023254
# 걸린 시간 :  2.734331369400024

# 시각화
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font = {'family': 'Calibri'}
plt.rc('font', **font)
plt.figure(figsize=(13, 10))
plt.plot(hist.history['loss'], marker='.', c='yellow', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='red', label='val_loss')
plt.title('loss & val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

## epochs 1000일 때, patience 100 일 때
# Epoch 448: early stopping
# loss :  0.011765165254473686
# acc :  1.0
# 걸린 시간 :  11.515390634536743
