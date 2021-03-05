# Analysing the data 
# Analysing the time taken by machine to change temperatures at different time of the day.


# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import glob
import datetime as dt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Importing the Dataset
file_names = glob.glob ('./results (6).csv')
k = 0

# Change these two variables to get different results
#degree change
n = 1
#operating state
op_state = 'Cooling'


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

# Visualising the result as time of day vs cooling time of machine
def plot (time_difference, y):
    fig = plt.figure (k)
    ax = plt.subplot ()
    plt.bar (y, time_difference, 0.05, align = 'center', alpha = 1)
    plt.xlabel ('Time of day')
    plt.ylabel ('Time taken for ' + op_state + ' ' + str (n) + ' Degree (in Minutes)')
    ax.set_xticks ([0,5,10,15,20,24])
    ax.set_xticklabels (['00:00', '05:00', '10:00', '15:00', '20:00', '00:00'])
    plt.title (os.path.basename (path))
    plt.show ()
    
    pp.savefig (fig)
    plt.close (fig)


# Finding time required for temparature change...
import json
def time_to_change_temperature (X):
    # Time to cool one degree in minutes
    time_difference = []
    # Factor represts temparature change between cooling/heating on and off.
    factor = 1
    y = []
    curr_temp = 0
    i = 0
    while i < len (X):
        content = json.loads (X.content[i])
        if 'status' in content and "room_temp" in content["status"]:
            curr_temp = content["status"]["room_temp"]
        
        if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in [op_state] :
            curr_index = i
            hh = tz_from_utc_ms_ts(X.created_date_time[i])
            while i < len (X):
                content = json.loads (X.content[i])
                if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in ["Off"]:
                    break
                i += 1
                
            if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in ["Off", "Fan_Only"]:
                j = i + 1
                while j < len (X):
                    content = json.loads (X.content[j])
                    if 'status' in content and "room_temp" in content["status"]:
                        factor = abs (curr_temp - content["status"]["room_temp"])
                        factor += 1
                        break
                    j += 1   
                diff = abs ((X.created_date_time[i] - X.created_date_time[curr_index])/(60000))
            
                if diff <= 1000 and factor == n:
                    time_difference.append (diff)
                    y.append (hh)
        i += 1
    if len (y) > 0:
        # plot (time_difference, y)
        print(time_difference)
        print(y)    



path = ''
pp = PdfPages ('./Plot' + str (n) + '.pdf')

for file_name in file_names:
    try:
      path = file_name
      dataset = pd.read_csv ((path))
    except:
      continue
    
    X = dataset.iloc[:, :]

    time_to_change_temperature (X)
    k += 1

pp.close ()