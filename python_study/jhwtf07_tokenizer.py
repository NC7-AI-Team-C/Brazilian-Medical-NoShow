import numpy as np
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

text = ' 나는 진짜 매우매우매우매우 매우 매우 매우 매우 맛있는 밥을 매우 엄청나게 많이 먹었고 배도 너무 엄청 엄청 엄청 엄청 엄청나게 배부르다.'

token = Tokenizer()
token.fit_on_texts([text])  # fit on 하면서 인덱스가 생성됌
# index = token.word_index 아래 print랑 같은 명령어
print(token.word_index)

# {'매우': 1, '엄청': 2, '엄청나게': 3, '나는': 4, '진짜': 5, 
#  '매우매우매우매우': 6, '맛있는': 7, '밥을': 8, '많이': 9, '먹었고': 10, 
#  '배도': 11, '너무': 12, '배부르 다': 13} ==> 가장 많이 나온 빈도에 따라 순서대로 생성됌

x = token.texts_to_sequences([text])
print(x)

## 원핫인코딩 하기 ## => 문장이 1개 일때 사용가능
from keras.utils import to_categorical

x = to_categorical(x)   # 원핫인코딩 하면 index + 1개가 수로 만들어짐 => 패딩을 위해 빈칸도 생성되기 때문에
print(x)
print(x.shape)  # (1, 21, 14) => 3차원으로 받아온다를 확인 가능

'''
####### One Hot Encoder 수정필요 ###### => 문장이 2개 이상 시 사용가능
from sklearn.preprocessing import OneHotEncoder 
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
x = x.reshape(-1, 11, 9)
onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape)

# 에러메시지 : AttributeError: 'list' object has no attribute 'reshape'
'''
