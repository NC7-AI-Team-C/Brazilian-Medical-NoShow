import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)    # KFold : 회귀모델 / StratifiedKFold : 분류모델

# Scaler 적용
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
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
    model = CatBoostRegressor(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = r2_score(CAT_model.predict(x_test), y_test)
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