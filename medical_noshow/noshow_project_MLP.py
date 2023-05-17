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
medical_noshow = pd.read_csv(path)

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

medical_noshow.AppointmentDay = pd.to_datetime(medical_noshow.AppointmentDay).dt.date
medical_noshow.ScheduledDay = pd.to_datetime(medical_noshow.ScheduledDay).dt.date
medical_noshow['PeriodBetween'] = medical_noshow.AppointmentDay - medical_noshow.ScheduledDay
# convert derived datetime to int
medical_noshow['PeriodBetween'] = medical_noshow['PeriodBetween'].dt.days

medical_noshow['diseaseCount'] = medical_noshow.Hipertension + medical_noshow.Diabetes + medical_noshow.Alcoholism

medical_noshow = medical_noshow.dropna(axis = 0)    # nan값을 가진 행 드랍

outliers = EllipticEnvelope(contamination=.1)      
# 이상치 탐지 모델 생성
outliers.fit(medical_noshow[['Age']])      
# 이상치 탐지 모델 훈련
predictions = outliers.predict(medical_noshow[['Age']])       
# 이상치 판별 결과
outlier_indices = np.where(predictions == -1)[0]    
# 이상치로 판별된 행의 인덱스를 추출
medical_noshow.loc[outlier_indices, 'Age']  = np.nan    #이상치를 nan처리
# 데이터프레임에서 이상치 행을 삭제
# print(medical_noshow[medical_noshow['PeriodBetween'] < 0])
medical_noshow[medical_noshow['PeriodBetween'] < 0] = np.nan    #이상치를 nan처리
medical_noshow = medical_noshow.fillna(np.nan)    #비어있는 데이터를 nan처리

medical_noshow = medical_noshow.dropna(axis = 0)    # nan값을 가진 행 드랍

# print(datasets.PeriodBetween.describe())
x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 
              'AppointmentDay', 'PeriodBetween', 'Age', 'Neighbourhood', 
              'Scholarship', 'diseaseCount', 'Hipertension', 'Diabetes', 'Alcoholism', 
              'Handcap', 'SMS_received']]
y = medical_noshow[['No-show']]

# print(x.info())
# print(y.info())

## 1-1. correlation hit map ##

# sns.set(font_scale = 1)
# sns.set(rc = {'figure.figsize':(12, 8)})
# sns.heatmap(data = medical_noshow.corr(), square = True, annot = True, cbar = True)
# plt.show()

## 1-2. drop useless data ##

x = x.drop(['PatientId', 'AppointmentID','ScheduledDay', 'Hipertension', 'Diabetes', 'Alcoholism'], axis=1)
print(x.describe())
# print(x.shape)

# print("이상치 정리 후\n",x.describe())
# print(x.shape, y.shape)

## 1-3. encoding object to int ##

encoder = LabelEncoder()

### char -> number
ob_col = list(x.dtypes[x.dtypes=='object'].index) ### data type이 object인 data들의 index를 ob_col 리스트에 저장
# print(ob_col)

for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)

# print(x.describe())
# print('y:', y[0:8], '...')
# print(x.info())

## 1-4. fill na data ##

print(x.describe())

## 1-5. check dataset ##

# print('head : \n',x.head(7))
# print('y : ',y[0:7]) ### y : np.array

##### 전처리 완료 #####



##### 훈련 구성 시작 #####
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=42
)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(1024, activation='relu'))
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
    filepath='./mcp/noshow01.hdf5'
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

