import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Data preprocessing #

path = './medical_noshow.csv'
medical_noshow = pd.read_csv(path)

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = medical_noshow[['No-show']]

# print(x.info())
# print(y.info())

## 1-1. correlation hit map ##

# sns.set(font_scale = 1)
# sns.set(rc = {'figure.figsize':(12, 8)})
# sns.heatmap(data = medical_noshow.corr(), square = True, annot = True, cbar = True)
# plt.show()

## 1-2. drop useless data ##

x = x.drop(['PatientId', 'AppointmentID'], axis=1)
# print(x.describe())

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
print(x.info())

## 1-4. fill na data ##

x = x.fillna(np.NaN)
# # print(x.describe())

## 1-5. check dataset ##

print('head : \n',x.head(7))
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

param = [
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

cb_model = CatBoostClassifier()
model = GridSearchCV(cb_model, param, cv=kfold, verbose=1, refit=True)

## 2-4. train model ##

start_time = time.time()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold )
acc = accuracy_score(y_test, y_predict)

end_time = time.time() - start_time

print('best params : ', model.best_params_)
print('best estimators : ', model.best_estimator_)
print('best score : ', model.best_score_) # 모델에서 나올 수 있는 최대의 스코어
print('model score : ', model.score(x_test, y_test)) # 테스트 데이터로 나오는 스코어
print('time : ', end_time)

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