import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.random.set_seed(77)
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = '../AI_study/'
datasets = pd.read_csv(path + 'medical_noshow.csv')

x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]

print(x.shape, y.shape)

for i in datasets.columns:
    print(i+":",len(datasets[i].unique()))

# 결측치 확인 ==> all False 나옴
print(datasets.isnull())          # 전체 데이터프레임의 결측치 여부 확인
print(datasets.isnull().any())    # 각 열에 대한 결측치 여부 확인
print(datasets.isnull().sum())    # 각 열의 결측치 개수 확인
print(datasets.isnull().any().any())  # 데이터프레임 전체에 결측치 여부 확인

datasets['PatientId'].astype('int64')   # PatientID 를 정수로 변환 (Dtype : float64 -> int64)
datasets.set_index('AppointmentID', inplace = True) # 데이터프레임을 효율적으로 사용하고 데이터를 쉽게 탐색할 수 있도록 인덱스로 설정
datasets['No-show'] = datasets['No-show'].replace({'No': 0, 'Yes': 1})  # 두 열을 변환하여 0 또는 1로 나타낸다
datasets['Gender'] = datasets['Gender'].replace({'F': 0, 'M': 1})

datasets['PreviousApp'] = datasets.groupby('PatientId').cumcount()  # 'PreviousApp' 열을 추가하여 각 환자의 이전 예약 수를 계산합니다.

# 'PreviousNoShow' 열을 추가하여 각 환자의 이전 예약에서의 결석 비율을 계산합니다.
#결측값은 0으로 채웁니다.
datasets['PreviousNoShow'] = datasets.groupby('PatientId')['No-show'].shift().expanding().mean().fillna(0)


datasets['PreviousNoShow'] = datasets['PreviousNoShow'].fillna(0)
datasets['PreviousNoShow']

datasets['Num_App_Missed'] = datasets.groupby('PatientId')['No-show'].cumsum()  # 열을 추가하여 각 환자의 누적 결석 횟수를 계산합니다.

#'ScheduledDay'와 'AppointmentDay'를 날짜 형식으로 변환합니다.
datasets['ScheduledDay'] = pd.to_datetime(datasets['ScheduledDay']).dt.date
datasets['AppointmentDay'] = pd.to_datetime(datasets['AppointmentDay']).dt.date

# 'Day_diff' 열을 추가하여 예약일과 실제 진료일 사이의 날짜 차이를 계산합니다.
datasets['Day_diff'] = (datasets['AppointmentDay'] - datasets['ScheduledDay']).dt.days
datasets['Day_diff'].unique()

datasets = datasets[(datasets.Age >= 0)]    # 'Age' 열의 값이 0이상인 행만 남기고 제거합니다. 즉, 음수값을 제거
datasets.drop(['ScheduledDay', 'AppointmentDay'], axis=1, inplace=True) # 여러 개의 열을 동시에 제거하고자 합니다.
datasets.drop(['PatientId', 'Neighbourhood'], axis=1,inplace = True)  # 한개의 열만을 지정해서 제거하고자 합니다.

#Convert to Categorical
datasets['Handcap'] = pd.Categorical(datasets['Handcap']) # 정수형 데이터를 범주형 데이터로 변환하기

#Convert to Dummy Variables
# 'pd.get_dummies()' 함수의 'columns' 매개변수에는 변환할 열을 지정하고
# 'prefix' 매개변수에는 생성될 더미 변수의 접두사를 지정합니다.
# 'Handcap'열의 값을 더미 변수로 변환하고, 새로운 더미 변수 열이 생성됩니다.
# 변환된 더미 변수는 'Handicap'이라는 접두사(prefix)를 갖는 열로 생성됩니다.
# 더미변수는 범주형 변수를 수치형으로 변환하는 방법 => 이진변수로 나타냄
datasets = pd.get_dummies(datasets, columns=['Handcap'], prefix='Handicap')

datasets = datasets[(datasets.Age >= 0) & (datasets.Age <= 100)]
datasets.info()

x = datasets.drop(['No-show'], axis=1)
y = datasets['No-show']

### kfold ###
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


## scaler ##
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x
# 최소-최대 스케일링은 데이터의 범위를 [0, 1] 또는 원하는 범위로 조정하는 방법입니다. 
# 이를 통해 각 특성 변수의 값들이 동일한 범위 내에 있도록 만들어줍니다.
# 이는 모델의 학습을 더욱 안정적으로 만들어줄 수 있습니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

### optuna ###
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def objectiveXGB(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'depth' : trial.suggest_int('depth', 8, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
        'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
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





