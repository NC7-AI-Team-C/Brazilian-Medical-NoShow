import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


path = './'
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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier # catboost로 돌린다.
from sklearn.ensemble import VotingClassifier 
# rd_clf = RandomForestClassifier()
# rd_clf.fit(X_train, y_train)


#####################kfold#########################
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

############################################모델 
xgb = XGBClassifier(
    n_estimators = 3502,
    learning_rate = 0.8716436036914981,
    random_state = 296
)
lgbm = LGBMClassifier(
    n_estimators = 3762,
    learning_rate = 0.34639249238719827,
    random_state = 1486
)
cat = CatBoostClassifier(
    n_estimators = 3964,
    depth = 16,
    fold_permutation_block = 94,
    learning_rate = 0.6568663313453064,
    od_pval = 0.05248246675064383,
    l2_leaf_reg = 1.336973598095982,
    random_state = 413
)

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='hard',
    n_jobs=1,
    verbose=0
)

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

classifiers = [cat, xgb, lgbm]
for model in classifiers:
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print(class_name, "'s score : ", score)

'''
CatBoostClassifier 's score :  0.9646217879116902
XGBClassifier 's score :  0.9643955845095911
LGBMClassifier 's score :  0.9646670285921101
'''