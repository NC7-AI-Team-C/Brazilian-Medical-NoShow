import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


path = 'C:/Users/bitcamp/Desktop/새 폴더/'
df = pd.read_csv(path + 'medical_noshow.csv')
print('Count of rows', str(df.shape[0]))
print('Count of Columns', str(df.shape[1]))

df.isnull().any().any()

for i in df.columns:
    print(i+":",len(df[i].unique()))


df['PatientId'].astype('int64')
df.set_index('AppointmentID', inplace = True)
df['No-show'] = df['No-show'].map({'No':0, 'Yes':1})
df['Gender'] = df['Gender'].map({'F':0, 'M':1})

df['PreviousApp'] = df.sort_values(by = ['PatientId','ScheduledDay']).groupby(['PatientId']).cumcount()
df['PreviousNoShow'] = (df[df['PreviousApp'] > 0].sort_values(['PatientId', 'ScheduledDay']).groupby(['PatientId'])['No-show'].cumsum() / df[df['PreviousApp'] > 0]['PreviousApp'])

df['PreviousNoShow'] = df['PreviousNoShow'].fillna(0)
df['PreviousNoShow']

# Number of Appointments Missed by Patient
df['Num_App_Missed'] = df.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())
df['Num_App_Missed']

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.strftime('%Y-%m-%d')
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['ScheduledDay']

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.strftime('%Y-%m-%d')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['AppointmentDay']

df['Day_diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Day_diff'].unique()

df = df[(df.Age >= 0)]
df.drop(['ScheduledDay'], axis=1, inplace=True)
df.drop(['AppointmentDay'], axis=1, inplace=True)
df.drop('PatientId', axis=1,inplace = True)
df.drop('Neighbourhood', axis=1,inplace = True)

#Convert to Categorical
df['Handcap'] = pd.Categorical(df['Handcap'])
#Convert to Dummy Variables
Handicap = pd.get_dummies(df['Handcap'], prefix = 'Handicap')
df = pd.concat([df, Handicap], axis=1)
df.drop(['Handcap'], axis=1, inplace = True)

df = df[(df.Age >= 0) & (df.Age <= 100)]
df.info()

X = df.drop(['No-show'], axis=1)
y = df['No-show']

#########################optuna_noshow##############################
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier # catboost로 돌린다. 

n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def objectiverandomforest(trial: Trial, X_train, y_train, X_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'random_state' : trial.suggest_int('random_state', 1, 2000)
    }
    # 학습 모델 생성
    model = RandomForestClassifier(**param)
    random_model = model.fit(X_train, y_train) # 학습 진행
    # 모델 성능 확인
    score = accuracy_score(random_model.predict(X_test), y_test)
    return score

# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

study.optimize(lambda trial : objectiverandomforest(trial, X, y, X_test), n_trials = 5)
print('Best trial : score {}, /nparams {}'.format(study.best_trial.value, 
                                                  study.best_trial.params))


print(optuna.visualization.plot_param_importances(study))
optuna.visualization.plot_optimization_history(study)
plt.show()



# [I 2023-05-17 20:19:59,241] A new study created in memory with name: no-name-4f02010d-1c0d-4396-afeb-3549833964f8
# [I 2023-05-17 20:21:23,898] Trial 0 finished with value: 0.9745747376040536 and parameters: {'n_estimators': 2865, 'max_depth': 15, 'random_state': 674}. Bef02010d-1c0d-4396-afeb-3549833964f8st is trial 0 with value: 0.9745747376040536.                                 d parameters: {'n_estimators': 2865, 'max_depth': 15, 'random_state': 674}. Best is trial 0 with value: 0.9745747376040536.
# [I 2023-05-17 20:22:40,518] Trial 1 finished with value: 0.9656170828809265 and parameters: {'n_estimators': 3291, 'max_depth': 8, 'random_state': 1211}. Best is trial 0 with value: 0.9745747376040536.
# [I 2023-05-17 20:23:37,191] Trial 2 finished with value: 0.9769272529858849 and parameters: {'n_estimators': 1810, 'max_depth': 16, 'random_state': 1705}. Best is trial 2 with value: 0.9769272529858849.
# [I 2023-05-17 20:24:12,343] Trial 3 finished with value: 0.9769724936663048 and parameters: {'n_estimators': 1167, 'max_depth': 16, 'random_state': 1603}. Best is trial 3 with value: 0.9769724936663048.
# [I 2023-05-17 20:25:18,170] Trial 4 finished with value: 0.9691458559536735 and parameters: {'n_estimators': 2465, 'max_depth': 12, 'random_state': 1844}. Best is trial 3 with value: 0.9769724936663048.
# Best trial : score 0.9769724936663048, /nparams {'n_estimators': 1167, 'max_depth': 16, 'random_state': 1603}
