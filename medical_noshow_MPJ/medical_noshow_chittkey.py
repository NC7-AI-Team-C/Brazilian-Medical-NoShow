import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '../AI_study/'
df = pd.read_csv(path + 'medical_noshow.csv')
print('Count of rows', str(df.shape[0]))    # 110527
print('Count of Columns', str(df.shape[1])) # 14

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
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)
clf_report = classification_report(y_test, y_pred_rd_clf)

print(f"Classification Report : \n{clf_report}")

from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(estimator = rd_clf, X = X, y =y, cv = 8)
print("avg acc: ",np.mean(accuracy))
print("acg std: ",np.std(accuracy))

acc = accuracy_score(y_test, y_pred_rd_clf)
print('cv pred acc : ', acc)

# avg acc:  0.9580432748755885
# acg std:  0.008782122369095046
# cv pred acc :  0.9596815056098443

# memory usage: 11.5 MB
# Classification Report : 
#               precision    recall  f1-score   support

#            0       0.98      0.97      0.97     22045
#            1       0.88      0.92      0.90      5585

#     accuracy                           0.96     27630
#    macro avg       0.93      0.94      0.94     27630
# weighted avg       0.96      0.96      0.96     27630

# avg acc:  0.9581156599642602
# acg std:  0.008852913829054032
