# coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import time

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# 1. Data preprocessing #

path = '../medical_noshow.csv'
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
##### 전처리 완료 #####

######### voting ############

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

xgb = XGBClassifier(
    subsample = 0.9, reg_lambda= 0.5, reg_alpha= 0.01, n_estimators= 500, 
    min_child_weight= 9, max_depth= 12, learning_rate= 0.01, gamma= 0.05, 
    colsample_bytree= 0.9, colsample_bynode= 0.9, colsample_bylevel= 0.3
) # optuna로 산출한 best_parameter 적용

lgbm = LGBMClassifier(
    subsample= 0.3, reg_lambda= 1.2, 
    reg_alpha= 0.05, num_leaves= 24, n_estimators= 700, min_data_in_leaf= 1, 
    min_child_samples= 90, max_depth= 3, learning_rate= 0.01, feature_fraction= 0.85, 
    colsample_bytree= 1.2
) # optuna로 산출한 best_parameter 적용

cat = CatBoostClassifier(
    subsample= 0.5, random_strength= 2, n_estimators= 500, learning_rate= 0.01, 
    l2_leaf_reg= 9, depth= 9, colsample_bylevel= 0.9, border_count= 10, 
    bagging_temperature= 1
) # optuna로 산출한 best_parameter 적용

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='hard',
    n_jobs=-1,
    verbose=0
)

model.fit(x_train, y_train)

y_voting_predict = model.predict(x_test)
voting_score = accuracy_score(y_test, y_voting_predict)

classifiers = [cat, xgb, lgbm]
for model in classifiers:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print(class_name, "'s score : ", score)

print('voting result : ', voting_score)
print('RandomSearch -> voting')



#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import time

from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# 1. Data preprocessing #

path = '../AI_study/'
df = pd.read_csv(path + 'medical_noshow.csv')
# CSV 파일을 읽어와서 DataFrame으로 저장

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

print('Count of rows', str(df.shape[0]))
print('Count of Columns', str(df.shape[1]))
# 데이터프레임의 크기와 열의 수를 출력

df.isnull().any().any()
# 데이터프레임의 전체에 빈 값이 있는지 여부를 확인

for i in df.columns:
    print(i+":",len(df[i].unique()))
# 데이터프레임의 각 열에 대해 유일한 값의 수를 출력
# 열 이름와 해당 열의 고유한 값의 개수가 함께 표시

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
# 'Handcap' 열을 범주형으로 변환하고, 더미 변수가 생성되고 기존의 'Handcap'열은 제거

df = df[(df.Age >= 0) & (df.Age <= 100)]
df.info()
# 'Age' 열의 값이 0 이상 100 이하인 행만 선택

x = df.drop(['No-show'], axis=1)
y = df['No-show']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# Min-Max 스케일링을 사용하여 특성 값을 0과 1 사이로 조정
##### 전처리 완료 #####


######### voting ############
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

xgb = XGBClassifier(
    colsample_bylevel=0.1, colsample_bynode=0.1, colsample_bytree=0.2,
    gamma=2, learning_rate=0.2, max_depth=3,
    min_child_weight=0.01, n_estimators=200, reg_alpha=0.1,
    reg_lambda=0.1, subsample=0.2
) # gridsearchCV로 산출한 best_parameter 적용

lgbm = LGBMClassifier(
    feature_fraction=0.2, learning_rate=0.1, max_depth=6,
    min_data_in_leaf=200, n_estimators=300, num_leaves=7,
    reg_alpha=0.1, reg_lambda=0.01
) # gridsearchCV로 산출한 best_parameter 적용

cat = CatBoostClassifier(
    colsample_bylevel=0.1,
    learning_rate=0.2, depth=3,
    n_estimators=200,
    reg_lambda=0.1, subsample=0.1
) # gridsearchCV로 산출한 best_parameter 적용

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='hard',
    n_jobs=-1,
    verbose=0
)

model.fit(x_train, y_train)

y_voting_predict = model.predict(x_test)
voting_score = accuracy_score(y_test, y_voting_predict)


classifiers = [cat, xgb, lgbm]
for model in classifiers:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print(class_name, "'s score : ", score)

print('voting result : ', voting_score)
print('GridSearchCV -> voting')


# CatBoostClassifier 's score :  0.9644860658704307
# XGBClassifier 's score :  0.9641693811074918
# [LightGBM] [Warning] feature_fraction is set=0.2, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.2
# [LightGBM] [Warning] min_data_in_leaf is set=200, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=200
# LGBMClassifier 's score :  0.9646217879116902
# voting result :  0.9645765472312704
# GridSearchCV -> voting