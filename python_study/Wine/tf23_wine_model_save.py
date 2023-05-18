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

model.save('./_save/tf16_wine.h5')



#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
## earlystopping
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=60,
                              mode='min', verbose=1, restore_best_weights=True)

start_time = time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=100,
               verbose=1, validation_split=0.2,
               callbacks=[earlystopping])
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

## epochs 350일 때, patience 100일 때
# Epoch 207: early stopping
# 걸린 시간 :  5.898505926132202
# loss :  0.5896075367927551
# acc :  0.8888888955116272

