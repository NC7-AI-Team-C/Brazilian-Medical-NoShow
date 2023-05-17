import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)



#############################전처리과정이후######################################
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier # catboost로 돌린다. 
# rd_clf = RandomForestClassifier()
# rd_clf.fit(X_train, y_train)

############################################모델 
model = CatBoostClassifier()




#####################kfold#########################
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)



# y_pred_rd_clf = rd_clf.predict(X_test)
# clf_report = classification_report(y_test, y_pred_rd_clf)

# print(f"Classification Report : \n{clf_report}")

# from sklearn.model_selection import cross_val_score

# accuracy = cross_val_score(estimator = rd_clf, X = X, y =y, cv = 8)
# print("avg acc: ",np.mean(accuracy))
# print("acg std: ",np.std(accuracy))

# from sklearn.model_selection import cross_val_score

# accuracy = cross_val_score(estimator = rd_clf, X = X, y =y, cv = 8)
# print("avg acc: ",np.mean(accuracy))
# print("acg std: ",np.std(accuracy))

#############예측####################### 
from sklearn.metrics import accuracy_score
import time
start_time = time.time()

model.fit(X_train, y_train)

result = model.score(X_test, y_test)
score = cross_val_score(model, X_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, X_test, y_test, cv=kfold )
acc = accuracy_score(y_test, y_predict)

end_time = time.time() - start_time

print('acc : ', acc)
print('소요시간 : ', end_time)


'''
캣부스트/하이퍼파라미터 디폴트입니다.
acc :  0.9620883098081795
소요시간 :  98.19628834724426
'''