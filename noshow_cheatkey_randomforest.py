import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time

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

########################################################################################
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

########################################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

########################################################################################
model = RandomForestClassifier(n_estimators = 1167, max_depth = 16, random_state = 1603)
n_splits = 10
random_state = 60
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#########################################################################################
start_time = time.time()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
score = cross_val_score(model, X_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, X_test, y_test, cv=kfold )
acc = accuracy_score(y_test, y_predict)

end_time = time.time() - start_time

print('acc : ', acc)
print('소요시간 : ', end_time)

## [I 2023-05-17 20:19:59,241] A new study created in memory with name: no-name-4f02010d-1c0d-4396-afeb-3549833964f8
# [I 2023-05-17 20:21:23,898] Trial 0 finished with value: 0.9745747376040536 and parameters: {'n_estimators': 2865, 'max_depth': 15, 'random_state': 674}. Bef02010d-1c0d-4396-afeb-3549833964f8st is trial 0 with value: 0.9745747376040536.                                 d parameters: {'n_estimators': 2865, 'max_depth': 15, 'random_state': 674}. Best is trial 0 with value: 0.9745747376040536.
# [I 2023-05-17 20:22:40,518] Trial 1 finished with value: 0.9656170828809265 and parameters: {'n_estimators': 3291, 'max_depth': 8, 'random_state': 1211}. Best is trial 0 with value: 0.9745747376040536.
# [I 2023-05-17 20:23:37,191] Trial 2 finished with value: 0.9769272529858849 and parameters: {'n_estimators': 1810, 'max_depth': 16, 'random_state': 1705}. Best is trial 2 with value: 0.9769272529858849.
# [I 2023-05-17 20:24:12,343] Trial 3 finished with value: 0.9769724936663048 and parameters: {'n_estimators': 1167, 'max_depth': 16, 'random_state': 1603}. Best is trial 3 with value: 0.9769724936663048.
# [I 2023-05-17 20:25:18,170] Trial 4 finished with value: 0.9691458559536735 and parameters: {'n_estimators': 2465, 'max_depth': 12, 'random_state': 1844}. Best is trial 3 with value: 0.9769724936663048.
# Best trial : score 0.9769724936663048, /nparams {'n_estimators': 1167, 'max_depth': 16, 'random_state': 1603}
# Best trial acc :  0.9630383640969961
# Best trial  소요시간 :  145.1654613018036

# 임의로 n_splits = 5 -> 11 , random_state =  42 -> 60  해본결과 조금 오릅니다 acc :  0.9638526963445531    소요시간 :  288.9823839664459