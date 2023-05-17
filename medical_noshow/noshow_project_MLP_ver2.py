#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# 1. Data preprocessing #

path = './medical_noshow.csv'
df = pd.read_csv(path)
# CSV 파일을 읽어와서 DataFrame으로 저장

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

print('Count of rows', str(df.shape[0]))
print('Count of Columns', str(df.shape[1]))
# 데이터프레임의 크기와 열의 수를 출력

df.isnull().any().any()

for i in df.columns:
    print(i+":",len(df[i].unique()))
# 데이터프레임의 각 열에 대해 유일한 값의 수를 출력

df['PatientId'].astype('int64')
df.set_index('AppointmentID', inplace = True)
# 'PatientId' 열을 정수형으로 변환하고, 'AppointmentID'를 인덱스로 설정
df['No-show'] = df['No-show'].map({'No':0, 'Yes':1})
# 'No-show' 열의 값('No', 'Yes')을 0과 1로 매핑
df['Gender'] = df['Gender'].map({'F':0, 'M':1})
# 'Gender' 열의 값('F', 'M')을 0과 1로 매핑

df['PreviousApp'] = df.sort_values(by = ['PatientId','ScheduledDay']).groupby(['PatientId']).cumcount()
# 'PreviousApp' 열을 생성하여 각 환자별 이전 약속 수를 계산
df['PreviousNoShow'] = (df[df['PreviousApp'] > 0].sort_values(['PatientId', 'ScheduledDay']).groupby(['PatientId'])['No-show'].cumsum() / df[df['PreviousApp'] > 0]['PreviousApp'])
# 'PreviousNoShow' 열을 생성하여 이전 약속에서의 No-show 비율을 계산

df['PreviousNoShow'] = df['PreviousNoShow'].fillna(0)
df['PreviousNoShow']
# 'PreviousNoShow' 열의 NaN 값을 0으로 채운다

# Number of Appointments Missed by Patient
df['Num_App_Missed'] = df.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())
df['Num_App_Missed']
# 'Num_App_Missed' 열을 생성하여 각 환자별 누적 No-show 수를 계산

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.strftime('%Y-%m-%d')
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['ScheduledDay']
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.strftime('%Y-%m-%d')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['AppointmentDay']
# 'ScheduledDay' 열과 'AppointmentDay' 열을 날짜 형식으로 변환

df['Day_diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
# 'Day_diff' 열을 생성하여 약속 일자와 예약 일자의 차이를 계산
df['Day_diff'].unique()
# Day_diff' 열의 유일한 값을 출력

df = df[(df.Age >= 0)]
# 'Age' 열의 값이 0 이상 행만 선택

df.drop(['ScheduledDay'], axis=1, inplace=True)
df.drop(['AppointmentDay'], axis=1, inplace=True)
df.drop('PatientId', axis=1,inplace = True)
df.drop('Neighbourhood', axis=1,inplace = True)
# 불필요한 열('ScheduledDay', 'AppointmentDay', 'PatientId', 'Neighbourhood')을 삭제

#Convert to Categorical
df['Handcap'] = pd.Categorical(df['Handcap'])
#Convert to Dummy Variables
Handicap = pd.get_dummies(df['Handcap'], prefix = 'Handicap')
df = pd.concat([df, Handicap], axis=1)
df.drop(['Handcap'], axis=1, inplace = True)
# 'Handcap' 열을 범주형으로 변환하고, 더미 변수로 변환

df = df[(df.Age >= 0) & (df.Age <= 100)]
df.info()
# 'Age' 열의 값이 0 이상 100 이하인 행만 선택

X = df.drop(['No-show'], axis=1)
y = df['No-show']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# Min-Max 스케일링을 사용하여 특성 값을 0과 1 사이로 조정

##### 전처리 완료 #####



##### 훈련 구성 시작 #####
x_train, x_test, y_train, y_test = train_test_split(
X, y, test_size = 0.2, shuffle=True, random_state=42
)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=16))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

##earlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              verbose=1, restore_best_weights=True ) # restore_best_weights의 기본값은 false이므로 true로 반드시 변경

# Model Check point
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./mcp/noshow_ver2_mcp01.hdf5'
)


start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32, 
          validation_split=0.2, 
          callbacks=[earlyStopping, mcp],
          verbose=1)
end_time = time.time() - start_time

loss, acc = model.evaluate(x_test, y_test)

model.summary()
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