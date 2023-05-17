import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor, VotingClassifier # voting 추가 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Data preprocessing #
path = '/Users/shinhyunwoo/Downloads/project/'
medical_noshow = pd.read_csv(path + 'medical_noshow.csv')
# path = './medical_noshow.csv'
# medical_noshow = pd.read_csv(path)

# SchduledDay와 AppointmentDay 값을 보면 날짜와 시간까지 나와있습니다. 시간까지 고려하면 좋겠지만 AppointmentDay의 시간은 모두 00:00:00이기 때문에 고려할 수가 없습니다. 따라서 날짜, 시간 타입을 날짜만 나와있는 타입으로 변경해서 사용할 겁니다.
medical_noshow.AppointmentDay = pd.to_datetime(medical_noshow.AppointmentDay).dt.date
medical_noshow.ScheduledDay = pd.to_datetime(medical_noshow.ScheduledDay).dt.date

# SchduledDay와 AppointmentDay 의 차이, 즉 예약을 한 시점으로부터 예약 날짜까지의 기간을 구해보겠습니다. 그 기간이 짧을수록 예약날에 진료를 갈 확률이 높을 것이라고 생각했습니다.
medical_noshow['PeriodBetween'] = medical_noshow.AppointmentDay - medical_noshow.ScheduledDay

# convert derived datetime to int
medical_noshow['PeriodBetween'] = medical_noshow['PeriodBetween'].dt.days
# print(medical_noshow.head())


# print(medical_noshow.columns)
# print(medical_noshow.head(10))

x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'PeriodBetween']]
y = medical_noshow[['No-show']]


# print(x.info())
# print(y.info())

## 1-1. correlation hit map ##

# sns.set(font_scale = 1)
# sns.set(rc = {'figure.figsize':(12, 8)})
# sns.heatmap(data = medical_noshow.corr(), square = True, annot = True, cbar = True)
# plt.show()

## 1-2. data preprocessing ##

x = x.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1)
print(x.describe())

medical_noshow[medical_noshow['PeriodBetween'] < 0] = 0
medical_noshow.replace({'Age': {-1: 0}}, inplace = True)
medical_noshow.drop(index = [63912, 63915, 68127, 76284, 97666], inplace = True)

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

x = x.fillna(np.NaN)
# # print(x.describe())


## 1-6. check dataset ##

print('head : ',x.head(7))
print('y : ',y[0:7]) ### y : np.array

# 2. Modeling #

## 2-1. Dividing into training and test data ##

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

## 2-2. scaling data & cross validating set ## 

n_splits = 21
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

## 2-3. create model ##

xgb = XGBClassifier()
lgbm = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='hard',
    n_jobs=-1, verbose=0
)

## 2-4. train model ##

# start_time = time.time()

model.fit(x_train, y_train)

# result = model.score(x_test, y_test)
# score = cross_val_score(model, x_train, y_train, cv=kfold)
# y_predict = cross_val_predict(model, x_test, y_test, cv=kfold )
# acc = accuracy_score(y_test, y_predict)

classfiers = [lgbm, xgb, cat]
for model in classfiers:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__ 

# end_time = time.time() - start_time
print('{0}, 정확도 : {1: .4f}'.format(class_name,score))


# print('acc : ', acc)
# print('소요시간 : ', end_time)

## 2-5. show feature importances

# print(model, " : ", model.feature_importances_) # sequential model has no attribute feature_importances
# n_features = x.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), ['Gender',	'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'])
# plt.title('noshow datset feature input importances')
# plt.ylabel('feature')
# plt.xlabel('importance')
# plt.ylim(-1, n_features)
# plt.show()

# LGBMClassifier, 정확도 :  0.7951
# XGBClassifier, 정확도 :  0.7932
# CatBoostClassifier, 정 