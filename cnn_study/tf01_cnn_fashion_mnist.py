import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
x_train, x_test = x_train/255.0, x_test/255.0

# Reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# [실습]

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, kernel_size=(4,4), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

'''
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, batch_size=256)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# MaxPooling result
# loss :  0.2244156450033188
# acc :  0.9215999841690063


# Maxpooling2D(2,2) 레이어 1, Dropout(0.3) result
# loss :  0.30810150504112244
# acc :  0.9257000088691711
'''