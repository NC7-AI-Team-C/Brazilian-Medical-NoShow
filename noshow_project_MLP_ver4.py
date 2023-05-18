#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import time
import warnings

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

# Data preprocessing #

path = './medical_noshow.csv'
df = pd.read_csv(path)
# CSV 파일을 읽어와서 DataFrame으로 저장

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

print('Count of rows', str(df.shape[0]))
print('Count of Columns', str(df.shape[1]))
# 데이터프레임의 크기와 칼럼의 수를 출력

df = df.fillna(np.nan)  # 결측값 nan으로 채우기

for column_name in df.columns:
    print(column_name+":",len(df[column_name].unique()))
# 데이터프레임의 각 칼럼에 대해 유일한 값의 수를 출력

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date
# 'ScheduledDay' 칼럼과 'AppointmentDay' 열을 날짜 형식으로 변환

df['Day_diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
# 'Day_diff' 칼럼을 생성하여 약속 일자와 예약 일자의 차이를 계산
print("Day_diff값 목록 \n",df['Day_diff'].unique())
# Day_diff' 칼럼의 유일한 값을 출력
print(df[df['Day_diff']<0]['No-show'])
# Day_diff<0 인 데이터는 노쇼로 처리했음을 확인

df.info()
print(df['No-show'][0:10])
# 각 컬럼의 데이터 타입 확인
ob_col = list(df.dtypes[df.dtypes=='object'].index) 
### data type이 object인 data들의 index를 ob_col 리스트에 저장
for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
# object인 데이터를 숫자형 데이터로 변환
df.info()
print(df['No-show'][0:10])
# [ no : 0, yes : 1 ]으로 정수화 되었음을 확인
# 각 컬럼의 데이터 타입 확인

df['PreviousApp'] = df.groupby(['PatientId']).cumcount()
# 'PatientId'가 같은 데이터끼리 그룹으로 묶어서 개수를 카운트 -> 각 환자별 이전 약속 수를 계산
# 이 값으로 'PreviousApp' 열을 생성

df['PreviousNoShow'] = (df[df['PreviousApp'] > 0].groupby(['PatientId'])['No-show'].cumsum() / df[df['PreviousApp'] > 0]['PreviousApp'])
# 이전에 예약한 기록이 있는 환자들만 선택,
# 'PatientId'가 같은 데이터끼리 그룹으로 묶어서 환자별로 고려,
# 이전에 noShow한 횟수의 합/이전에 예약한 횟수 = 해당 환자의 noShow 비율 계산
# 'PreviousNoShow' 칼럼을 생성하여 이전 약속에서의 No-show 비율 칼럼 생성

df['PreviousNoShow'] = df['PreviousNoShow'].fillna(0)
# 'PreviousNoShow' 칼럼의 NaN 값을 0으로 채운다. 
# 즉, 첫 예약자는 이전에 noShow 안한것으로 간주

# Number of Appointments Missed by Patient
df['Num_App_Missed'] = df.groupby('PatientId')['No-show'].cumsum()
# 'PatientId'가 같은 데이터끼리 그룹으로 묶어서 환자별로 고려,
# 'Num_App_Missed' 각 환자별 누적 No-show 수를 계산한 칼럼을 생성


# print("handcap 종류 : ",df['Handcap'].unique())
df['Handcap'] = pd.Categorical(df['Handcap'])
#  원 핫 인코딩 개념을 이용해서 핸드캡을 범주형 데이터로 변환
Handicap = pd.get_dummies(df['Handcap'], prefix = 'Handicap')
# 핸드캡 칼럼을 핸디캡 더미 변수로 변환(변수명에 i만 추가됨 주의)
# prefix='Handicap'는 생성된 더미 변수의 이름에 'Handicap' 접두사를 붙이도록 지정
df = pd.concat([df, Handicap], axis=1)
# 데이터 프레임에 핸디캡 변수를 추가, 데이터 프레임을 열방향으로 병합
df.drop(['Handcap','ScheduledDay','AppointmentDay', 'AppointmentID','PatientId','Neighbourhood'], axis=1, inplace=True)
# 불필요한 칼럼 삭제, inplace=True 파라미터를 통해 원본 데이터프레임 수정
print(df.describe())

df = df[(df.Age >= 0) & (df.Age <= 100)]
df.info()
# 'Age' 열의 값이 0 이상 100 이하인 행만 선택 # 이외의 값은 이상치로 판정

x = df.drop(['No-show'], axis=1)
y = df['No-show']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# Min-Max 스케일링을 사용하여 특성 값을 0과 1 사이로 조정

##### Complete Data Preprocessing #####

# 2. modeling #

## 2-1. split train data and test data ## 

x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=42
)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

## 2-2. create model ##

model = Sequential()
model.add(Dense(16, input_dim=16, activation='linear'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

## 2-3. train model ##

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

## EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=25, mode='min',
                              verbose=1, restore_best_weights=True ) # restore_best_weights의 기본값은 false이므로 true로 반드시 변경

# Model Check point
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./medical_noshow/mcp/noshow_ver5_layer2_bat32.hdf5'
    ######################################
    # 훈련전에 mcp파일 명 변경 잊지 말기!! #
    ######################################
)

<<<<<<< HEAD:medical_noshow/noshow_project_MLP_ver4.py
batch_size=32
=======
>>>>>>> 5b274416ac00897ceb4fe130b49bcbbc4a42edac:noshow_project_MLP_ver4.py
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=batch_size, 
          validation_split=0.2, 
          callbacks=[earlyStopping, mcp],
          verbose=1)
end_time = time.time() - start_time

loss, acc = model.evaluate(x_test, y_test)

model.summary()
print('batch_size : ', batch_size)
print('loss : ', loss)
print('acc : ', acc)
print('소요시간 : ', end_time)

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_4 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_5 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_6 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 5,073
# Trainable params: 5,073
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.08760078996419907
# acc :  0.9627217054367065
# 소요시간 :  1270.6346561908722

# ver2 레이어 수를 줄였음에도 불구하고 훈련시간이 길어지고 결과가 나빠짐
# 레이어를 줄인만큼 에포크 수가 늘어났고 모델이 단순화되어 성능이 떨어지는것으로 보임
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 33        
# =================================================================
# Total params: 2,961
# Trainable params: 2,961
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.0904354602098465
# acc :  0.9614549279212952
# 소요시간 :  1427.7385559082031

# 노드 수를 줄이고 레이어 수를 높여도 정확도 하락
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_2 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_3 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_4 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_5 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_6 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_7 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_8 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_9 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_10 (Dense)             (None, 1)                 17
# =================================================================
# Total params: 2,737
# Trainable params: 2,737
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.08681493252515793
# acc :  0.9618168473243713
# 소요시간 :  2917.160630464554

# 전처리 코드 수정판 -> acc 값 상승
# Epoch 00080: val_loss did not improve from 0.05412
# 2211/2211 [==============================] - 9s 4ms/step - loss: 0.0485 - accuracy: 0.9749 - val_loss: 0.0589 - val_accuracy: 0.9709
# Epoch 00080: early stopping
# 691/691 [==============================] - 2s 2ms/step - loss: 0.0523 - accuracy: 0.9724
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 2,961
# Trainable params: 2,961
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.05232706665992737
# acc :  0.9724484086036682
# 소요시간 :  725.5225591659546

###########################################################################
# ver4
# 레이어 갯수에 따라 결과 비교하기
# Epoch 00106: val_loss did not improve from 0.05498
# 1106/1106 [==============================] - 4s 4ms/step - loss: 0.0505 - accuracy: 0.9733 - val_loss: 0.0553 - val_accuracy: 0.9720
# Epoch 00106: early stopping
# 691/691 [==============================] - 2s 2ms/step - loss: 0.0515 - accuracy: 0.9720
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 849
# Trainable params: 849
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.05149100348353386
# acc :  0.971996009349823  # 은닉층 1개
# 소요시간 :  478.8084788322449

# Epoch 00071: val_loss did not improve from 0.05350
# 1106/1106 [==============================] - 4s 4ms/step - loss: 0.0503 - accuracy: 0.9734 - val_loss: 0.0551 - val_accuracy: 0.9721
# Epoch 00071: early stopping
# 691/691 [==============================] - 2s 3ms/step - loss: 0.0522 - accuracy: 0.9725
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272       
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 1,905
# Trainable params: 1,905
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.052245061844587326
# acc :  0.9724936485290527 # 은닉층 2개
# 소요시간 :  345.04135298728943

# Epoch 00062: val_loss did not improve from 0.05453
# 1106/1106 [==============================] - 5s 4ms/step - loss: 0.0503 - accuracy: 0.9733 - val_loss: 0.0553 - val_accuracy: 0.9713
# Epoch 00062: early stopping
# 691/691 [==============================] - 2s 3ms/step - loss: 0.0520 - accuracy: 0.9724
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 2,961
# Trainable params: 2,961
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.052001629024744034
# acc :  0.9724484086036682 # 은닉층 3개 # 2개에 비해 0.00005 하락 # 유의미한 차이는 아니라 노드 들려가며 비교 필요
# 소요시간 :  304.74008560180664

# Epoch 00064: val_loss did not improve from 0.05441
# 1106/1106 [==============================] - 4s 4ms/step - loss: 0.0506 - accuracy: 0.9732 - val_loss: 0.0560 - val_accuracy: 0.9721
# Epoch 00064: early stopping
# 691/691 [==============================] - 2s 3ms/step - loss: 0.0517 - accuracy: 0.9724
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 1,905
# Trainable params: 1,905
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.05166608840227127
# acc :  0.9724484086036682 # 은닉층 2개 입력층에 relu 적용 #은닉층 3개와 같은 결과값
# 소요시간 :  299.9599003791809

# Epoch 00158: val_loss did not improve from 0.05313
# 1106/1106 [==============================] - 4s 4ms/step - loss: 0.0509 - accuracy: 0.9730 - val_loss: 0.0547 - val_accuracy: 0.9729
# Epoch 00158: early stopping
# 691/691 [==============================] - 2s 3ms/step - loss: 0.0518 - accuracy: 0.9722
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 1,905
# Trainable params: 1,905
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.051755670458078384
# acc :  0.9721769690513611
# 소요시간 :  739.606773853302

# Epoch 00158: val_loss did not improve from 0.05313
# 1106/1106 [==============================] - 4s 4ms/step - loss: 0.0509 - accuracy: 0.9730 - val_loss: 0.0547 - val_accuracy: 0.9729
# Epoch 00158: early stopping
# 691/691 [==============================] - 2s 3ms/step - loss: 0.0518 - accuracy: 0.9722
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                272
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 1,905
# Trainable params: 1,905
# Non-trainable params: 0
# _________________________________________________________________
# loss :  0.051755670458078384
# acc :  0.9721769690513611 # 입력층에 시그모이드 적용 -> 결과 하락 (디폴트는 linear)
# 소요시간 :  739.606773853302