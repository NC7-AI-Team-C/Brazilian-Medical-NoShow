import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

path = './medical_noshow.csv'
medical_noshow = pd.read_csv(path)

x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = medical_noshow[['No-show']]

x = x.drop(['PatientId', 'AppointmentID'], axis=1)

encoder = LabelEncoder()

ob_col = list(x.dtypes[x.dtypes=='object'].index) ### data type이 object인 data들의 index를 ob_col 리스트에 저장

for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)

# print(x.info())

x = x.fillna(np.NaN)

print(x['Gender'].values) # Female = 0, Male = 1
# print(x['ScheduledDay'])
# print(x['AppointmentDay'])
# print(x['Age'])
# print(x['Neighbourhood'])
# print(x['Scholarship'])
# print(x['Hipertension'])
# print(x['Diabetes'])
# print(x['Alcoholism'])
# print(x['Handcap'])
# print(x['SMS_received'])


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print('1 사분위 : ', quartile_1)
    print('2 사분위 : ', q2)
    print('3 사분위 : ', quartile_3)

    iqr = quartile_3 - quartile_1
    print('iqr : ', iqr)

    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)

    print('lower_bound : ', lower_bound)
    print('upper_bound : ', upper_bound)

    return np.where((data_out > upper_bound) | (data_out < lower_bound))

column_name = 'SMS_received'

oliers = np.array(x[column_name].values)

outliers_loc = outliers(oliers)

print('이상치의 위치 : ', outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(oliers)
plt.title('%s' % column_name)
plt.ylabel('Y')
plt.xlabel('X')
plt.show()