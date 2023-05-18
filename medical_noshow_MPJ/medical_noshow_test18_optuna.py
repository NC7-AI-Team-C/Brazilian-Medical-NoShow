import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
path = '../AI_study/'
datasets = pd.read_csv(path + 'medical_noshow.csv')

print('columns : \n',datasets.columns)
print('head : \n',datasets.head(7))

x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]
print(x.shape, y.shape)

print(x.info())  # info() 컬럼명, null값, data타입 확인

print(x.describe())

# 결과에 영향을 주지 않는 값 삭제
x = x.drop(['PatientId', 'AppointmentID'], axis=1)

print(x.shape)

x = x.fillna(np.NaN)

# 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index)    # 오브젝트 컬럼 리스트 추출
print(ob_col)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)
# no = 0 , yes = 1

x = x.fillna(np.NaN)

print('columns : \n',x.columns)
print('head : \n',x.head(7))
print('y : ',y[0:8])

x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=62)

### StratifiedKFold ###
n_splits = 35
random_state = 62
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Scaler 적용
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def objectiveCAT(trial: Trial, x_train, y_train, x_test):
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
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    return score

# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 5)
print('Best trial : score {}, /nparams {}'.format(study.best_trial.value, 
                                                  study.best_trial.params))

# 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
print(optuna.visualization.plot_param_importances(study))
# 하이퍼파라미터 최적화 과정을 확인
optuna.visualization.plot_optimization_history(study)
plt.show()