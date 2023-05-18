import numpy as np
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

text1 = ' 나는 진짜 매우매우매우매우 매우 매우 매우 매우 맛있는 밥을 매우 엄청나게 많이 먹었고 배도 너무 엄청 엄청 엄청 엄청 엄청나게 배부르다.'
text2 = '나는 딥러닝이 정말 매우 너무 엄청 재미있다. 재미있어 하는 내가 매우 매우 엄청 너무 너무 너무 너무 너무 너무 멋있다. 또 또 또 얘기해봐바'

token = Tokenizer()
token.fit_on_texts([text1, text2])  # fit on 하면서 인덱스가 생성됌
# index = token.word_index 아래 print랑 같은 명령어

print(token.word_index)

# {'매우': 1, '너무': 2, '엄청': 3, '또': 4, '나는': 5, '엄청나게': 6, '진짜': 7, '
# 매우매우매우매우': 8, '맛있는': 9, '밥을': 10, '많이': 11, '먹었고': 12, '배도': 13, '배부르다': 14, '딥러닝이': 15, '정말': 16, '재미있다': 17, '재미있어': 18, '하
# 는': 19, '내가': 20, '멋있다': 21, '얘기해봐바': 22}

x = token.texts_to_sequences([text1, text2])
print(x)
# text 1 :[[5, 7, 8, 1, 1, 1, 1, 9, 10, 1, 6, 11, 12, 13, 2, 3, 3, 3, 3, 6, 14]
# text 2 : [5, 15, 16, 1, 2, 3, 17, 18, 19, 20, 1, 1, 3, 2, 2, 2, 2, 2, 2, 21, 4, 4, 4, 22]]

## 원핫인코딩 하기 ## => 문장이 1개 일때 사용가능
from keras.utils import to_categorical

x_new = x[0] + x[1]
print(x_new)
# [5, 7, 8, 1, 1, 1, 1, 9, 10, 1, 6, 11, 12, 13, 2, 3, 3, 3, 3, 6, 14, 5, 
# 15, 16, 1, 2, 3, 17, 18, 19, 20, 1, 1, 3, 2, 2, 2, 2, 2, 2, 21, 4, 4, 
# 4, 22]

x = to_categorical(x_new)   # 원핫인코딩 하면 index + 1개가 수로 만들어짐 => 패딩을 위해 빈칸도 생성되기 때문에
print(x)
print(x.shape)  # (45, 23) ==> shape를 노여야 함! 3차원으로 원핫인코딩을 통해서

####### One Hot Encoder 수정필요 ###### => 문장이 2개 이상 시 사용가능
from sklearn.preprocessing import OneHotEncoder 
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
# x = x.reshape(-1, 11, 9)
x = np.array(x_new)
print(x)
print(x.shape)  # (45,)
x = x.reshape(-1, 1)    # 차원이 1개 늘어남
print(x.shape)  # (45, 1)
onehot_encoder.fit(x)    # 순서가 중요!!
x = onehot_encoder.transform(x)
print(x)
print(x.shape)  # (45, 22)