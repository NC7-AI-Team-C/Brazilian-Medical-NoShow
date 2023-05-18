import numpy as np
from keras.preprocessing.text import Tokenizer


#1. 데이터
docs = ['재밌어요', '재미없다', '돈 아깝다', '숙면했어요',
        '최고에요', '꼭 봐라', '세 번 봐라', '또 보고싶다',
        'n회차 관람', '배우가 잘 생기긴 했어요', '발연기에요', '최악이에요',
        '후회된다', '돈 버렸다', '글쎄요', '보다 나왔다',
        '망작이다', '연기가 어색해요', '차라리 기부할걸', '후속편이 기대되요', 
        '감동적이에요', '다른 거 볼걸 그랬어요', '같이 보면 더 재밌어요']
#23
# 긍정 1, 부정 0
labels = np.array([1, 0, 0, 0,
                   1, 1, 1, 1,
                   1, 1, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   1, 0, 1])

#Tokenizer
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_docs)

x = token.texts_to_sequences(docs)
print(x)
# [[1], [4], [2, 5], [6], [7], [8, 3], [9, 10, 3], [11, 12], [13, 14],
# [15, 16, 17, 18], [19], [20], [21], [2, 22], [23], [24, 25], [26],
# [27, 28], [29, 30], [31, 32], [33], [34, 35, 36, 37], [38, 39, 40, 1]]

### pad_sequences ###
from keras_preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=4)
print(pad_x)
print(pad_x.shape)  # (23, 4) => 4는 maxlen

word_size = len(token.word_index)
print('word_size = ' , word_size)   # word_size =  40

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim = 41, output_dim = 10, input_length = 4))    #input_dim은 워드사이즈 + 1 / out_dim은 node 수 / input_length는 문장으 길이
# model.add(Embedding)
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))   # 이진 분류
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(pad_x, labels, epochs=100, batch_size=16)

#4. 평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('accuracy : ', acc)

# loss :  0.06543272733688354
# accuracy :  1.0


