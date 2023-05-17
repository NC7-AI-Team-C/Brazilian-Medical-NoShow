import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Data preprocessing #

path = './medical_noshow.csv'
medical_noshow = pd.read_csv(path)

# convert datetime of AppointmentDay, ScheduledDay to date
medical_noshow.AppointmentDay = pd.to_datetime(medical_noshow.AppointmentDay).dt.date
medical_noshow.ScheduledDay = pd.to_datetime(medical_noshow.ScheduledDay).dt.date

# create new column named 'PeriodBetween', which is difference between AppointmentDay and ScheduledDay
medical_noshow['PeriodBetween'] = medical_noshow.AppointmentDay - medical_noshow.ScheduledDay

# convert derived datetime to int
medical_noshow['PeriodBetween'] = medical_noshow['PeriodBetween'].dt.days
# print(medical_noshow.head())

# Combine all disease columns into one column 'Diseases'
medical_noshow['Diseases'] = medical_noshow['Diabetes'] + medical_noshow['Hipertension'] + medical_noshow['Alcoholism']
# print(medical_noshow['Diseases'])

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'PeriodBetween', 'Diseases']]
y = medical_noshow[['No-show']]

# print(x.info())
# print(y.info())

## 1-1. correlation hit map ##

# sns.set(font_scale = 1)
# sns.set(rc = {'figure.figsize':(12, 8)})
# sns.heatmap(data = medical_noshow.corr(), square = True, annot = True, cbar = True)
# plt.show()

## 1-2. data preprocessing ##

x = x.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood', 'Diabetes', 'Hipertension', 'Alcoholism'], axis=1)
print(x.describe())

medical_noshow[medical_noshow['PeriodBetween'] < 0] = 0 # convert negative period to zero
medical_noshow.replace({'Age': {-1: 0}}, inplace = True)
medical_noshow.drop(index = [63912, 63915, 68127, 76284, 97666], inplace = True) # drop outliers of 'Age'

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

## 1-5. check dataset ##

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

model = XGBClassifier(
    random_state=123,
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    gamma=1
)

## 2-4. train model ##

start_time = time.time()

model.fit(
    x_train, y_train, early_stopping_rounds=20,
    eval_set=[(x_train, y_train), (x_test, y_test)],
    eval_metric='error'
    # regression model : rmse, mae, rmsle...
    # binary classification : error, auc, logloss...
    # multiclass classification : merror, mlogloss...
)

result = model.score(x_test, y_test)
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold )
acc = accuracy_score(y_test, y_predict)

end_time = time.time() - start_time

print('acc : ', acc)
print('소요시간 : ', end_time)

# SelectFromModel
thresholds = model.feature_importances_

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, acc:%.2f%%" % (thresh, select_x_train.shape[1], score*100))

'''
acc :  0.7938116348502668
소요시간 :  200.53839492797852
(66316, 5) (22106, 5)
Thresh=0.050, n=5, acc:79.34%
(66316, 3) (22106, 3)
Thresh=0.084, n=3, acc:79.25%
(66316, 4) (22106, 4)
Thresh=0.050, n=4, acc:79.25%
(66316, 6) (22106, 6)
Thresh=0.042, n=6, acc:79.28%
(66316, 2) (22106, 2)
Thresh=0.101, n=2, acc:79.54%
(66316, 1) (22106, 1)
Thresh=0.638, n=1, acc:79.53%
(66316, 7) (22106, 7)
Thresh=0.035, n=7, acc:79.20%
'''

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