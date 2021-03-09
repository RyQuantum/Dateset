# -*- coding: utf-8 -*-
"""
Author: Suvab

"""


import pandas as pd
import glob
import datetime as dt
import json
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt




maximum_temp_change = []
current_max = 0
#operating state
op_state = 'Heating'
exclude_state = 'Cooling'
required_data = {
        'TIME_OF_DAY': [],
        'TEMP_CHANGE': [],
        'TIME_REQUIRED': [],
        }



# Given millisecond utc timestamp return time
def tz_from_utc_ms_ts(utc_ms_ts):
    
    # convert from time stamp to datetime
    utc_datetime = dt.datetime.utcfromtimestamp(utc_ms_ts / 1000.)
    
    # getting time from datetime object
    res = utc_datetime.hour + utc_datetime.minute / 60
    if res - 5 < 0:
        res = 24 + (res - 5)
    else:
        res = res - 5    
    return res



#Preprocess the existing logs


def preprocessData (X, temp_change, isCounter = False):
    time_difference = []
    # change represents temparature change between cooling/heating on and off.
    change = 1
    y = []
    curr_temp = 0
    i = 0
    global maximum_temp_change
    global current_max
    while i < len(X):

        content = json.loads (X.content[i])
        if 'status' in content and "room_temp" in content["status"]:
            curr_temp = content["status"]["room_temp"]
        
        if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in [op_state] :
            curr_index = i
            
            time_of_day = tz_from_utc_ms_ts(X.created_date_time[i])
            while i < len(X):
                content = json.loads (X.content[i])
                if 'status' in content and "room_temp" in content["status"]:
                    if(op_state == "Cooling"):
                        change = curr_temp - content["status"]["room_temp"]
                    else:
                        change =  content["status"]["room_temp"] - curr_temp
                    
                    if change > current_max:
                        maximum_temp_change.append(change)
                        current_max = change
                    if( change == temp_change):
                      diff = abs((X.created_date_time[i] - X.created_date_time[curr_index])/(60000))
                      time_difference.append (diff)
                      y.append(time_of_day)
                      if(isCounter == False):
                        required_data['TIME_OF_DAY'].append(time_of_day)
                        required_data['TEMP_CHANGE'].append(temp_change)
                        required_data['TIME_REQUIRED'].append(diff)
                      
                      break
                if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in ["Off", exclude_state, "Fan_Only"]:
                    break
                i=i+1

        i=i+1       
    if (isCounter):
        return maximum_temp_change
    return required_data



# Importing the Dataset
path = glob.glob ('./EthernetData/dataset6.csv')
dataset = pd.read_csv ((path[0]))
    
X = dataset.iloc[:, :]
X_filtered = ''
total_temp_changes = preprocessData(X, 0, True)
for temp_changes in total_temp_changes:
    X_filtered = preprocessData(X, temp_changes)
X_filtered_dataframe = pd.DataFrame(X_filtered)
X_filtered_dataframe.drop(['TIME_OF_DAY'], axis=1, inplace=True)  
plt.scatter(X_filtered_dataframe['TEMP_CHANGE'], X_filtered_dataframe['TIME_REQUIRED'])
plt.show()
X_train, X_test = train_test_split(
    X_filtered_dataframe.values, test_size=0.20, random_state=42)


#NN Model (dot(input, weight) + bias)
model = Sequential()
model.add(Dense(50, input_dim=X_filtered_dataframe.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(X_filtered_dataframe.shape[1]))
# optimizer = Adam(lr = 0.01)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(X_train,X_train,verbose=1,epochs=80)

pred_test = model.predict(X_test)
max_loss_mae_test = (np.mean(np.abs(pred_test - X_test), axis=1))
rmse_test = np.sqrt(metrics.mean_squared_error(pred_test,X_test))
# print('MAE loss from test dataset',max_loss_mae_test)
# print('RMSE loss from test dataset', rmse_test)


pred_train = model.predict(X_train)
max_loss_mae_train = max(np.mean(np.abs(pred_train - X_train), axis=1))
rmse_train = np.sqrt(metrics.mean_squared_error(pred_train,X_train))
print('MAX MAE loss from train dataset',max_loss_mae_train)

X_bad  = {
    'TEMP_CHANGE': [7.], 'TIME_REQUIRED': [80.]}
X_bad_dataframe = pd.DataFrame(X_bad)
X_bad_values = X_bad_dataframe.values

pred_false = model.predict(X_bad_values)
rmse_false_custom = np.sqrt(metrics.mean_squared_error(pred_false,X_bad_values))
mae_false_custom = (np.mean(np.abs(pred_false - X_bad_values), axis=1))
# print('RMSE for train dataset', rmse_train)
# print('RMSE for custom value', rmse_false_custom)
print('MAE for custom value', mae_false_custom)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")




