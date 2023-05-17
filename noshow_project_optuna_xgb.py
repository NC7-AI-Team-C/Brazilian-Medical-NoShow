import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import optuna

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

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

def objectiveXGB(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
        'random_state' :trial.suggest_int('random_state', 1, 2000)
    }
    # 학습 모델 생성
    model = XGBClassifier(**param)
    XGB_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = accuracy_score(XGB_model.predict(x_test), y_test)
    return score

# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveXGB(trial, x, y, x_test), n_trials = 5)
print('Best trial : score {}, /nparams {}'.format(study.best_trial.value, 
                                                  study.best_trial.params))

# 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
print(optuna.visualization.plot_param_importances(study))
# 하이퍼파라미터 최적화 과정을 확인
optuna.visualization.plot_optimization_history(study)
plt.show()

'''
    n_estimators = 951,
    learning_rate = 0.7979977093999802,
    random_state = 1760 
'''