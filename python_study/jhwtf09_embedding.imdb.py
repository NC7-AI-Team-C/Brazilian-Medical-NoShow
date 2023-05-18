import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping   # keras 에서 EarlyStopping 가져오기
from sklearn.metrics import r2_score
# from keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 10000
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(np.unique(y_train, return_counts=True))
print(len(np.unique(y_train)))  # 2

# 최대길이와 평균길이
print('리뷰의 최대길이 : ', max(len(i) for i in x_train))
print('리뷰의 평균길이 : ', sum(map(len, x_train)) / len(x_train))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# pad_sequences
x_train = pad_sequences(x_train, padding='pre',
                       maxlen=100)
x_test = pad_sequences(x_test, padding='pre',
                      maxlen=100)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim = 100))
model.add(LSTM(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

# Total params: 2,419,906
# Trainable params: 2,419,906
# Non-trainable params: 0

## [실습] 코드 완성하기!

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics='accuracy')
earlystopping = EarlyStopping(monitor='val_loss', patience=100,
                              mode='min', restore_best_weights=True)
start_time = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print('loss : ', loss)
print('acc : ', acc)
print('r2 스코어 : ', r2_score)
print('걸린 시간 : ', end_time)






