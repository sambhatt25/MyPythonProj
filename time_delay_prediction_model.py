import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('train_data.csv')

df_f=df[(df.timeTableRowStationShortCode=="TPE") & (df.timeTableRowType =="ARRIVAL")] # considering Tampere (TPE) arival time
df_f['timeTableRowActualTime']=pd.to_datetime(df_f.timeTableRowActualTime, format="%Y-%m-%dT%H:%M:%S.%fZ") # converting into timestamp
df_f['timeTableRowScheduledTime']=pd.to_datetime(df_f.timeTableRowScheduledTime, format="%Y-%m-%dT%H:%M:%S.%fZ")
df_f['time_delay']= df_f['timeTableRowActualTime'] - df_f['timeTableRowScheduledTime']
df_f['time_delay_in_seconds']=df_f['time_delay']/np.timedelta64(1,'s')    #calculating delay in seconds

data_1=df_f.filter(items=['time_delay_in_seconds'])
data_1['sl']=data_1.index #adding a linear index variable to make the model

data=data_1.filter(items=['sl','time_delay_in_seconds']).dropna() # droping unavailable data

X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

print(Y_pred)
