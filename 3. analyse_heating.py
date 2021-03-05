# Analysing the data 
# Analysing daily behavior of thermostat (heating).


# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import glob
import datetime as dt
import json
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

# Importing the Dataset
file_names = glob.glob ('../EthernetData/Hub*')
k = 0


# Given millisecond utc timestamp year
def find_year(utc_ms_ts):
    
    # convert from time stamp to datetime
    res = dt.datetime.utcfromtimestamp(utc_ms_ts / 1000.)
    return res.year


# Given millisecond utc timestamp return month
def find_month(utc_ms_ts):
    
    # convert from time stamp to datetime
    res = dt.datetime.utcfromtimestamp(utc_ms_ts / 1000.)
    return res.month

# Given millisecond utc timestamp return day 
def find_day (utc_ms_ts):
    
    res = dt.datetime.utcfromtimestamp(utc_ms_ts / 1000.)
    return res.day

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
    

# Gettting data for plotting.
    
def get_plot_data (X):
    i = 0
    while i < len (X):
        content = json.loads (X.content[i])
        if 'status' in content and "room_temp" in content["status"]:
            room_temp.append (content['status']['room_temp'])
            timestamp1.append (tz_from_utc_ms_ts(X.created_date_time[i]))
            month1.append (find_month (X.created_date_time[i]))
            day1.append (find_day (X.created_date_time[i]))
            year1.append (find_year (X.created_date_time[i]))
            
        if 'status' in content and "heating_setpoint" in content["status"]:
            heatsetpoint.append (content['status']['heating_setpoint'])
            timestamp2.append (tz_from_utc_ms_ts(X.created_date_time[i]))
            month2.append (find_month (X.created_date_time[i]))
            day2.append (find_day (X.created_date_time[i]))
            year2.append (find_year (X.created_date_time[i]))
                        
        i += 1
        
    i = 0
    while i < len (X):
        content = json.loads (X.content[i])
        if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in ["Heating"] :
             while i < len (X):
                 content = json.loads (X.content[i])
                 if 'status' in content and "operating_state" in content["status"] and content["status"]["operating_state"] in ["Off", 'Fan_Only'] :
                     break
                 if 'status' in content and "room_temp" in content["status"]:
                     heating.append (content['status']['room_temp'])
                     timestamp3.append (tz_from_utc_ms_ts(X.created_date_time[i]))
                     month3.append (find_month (X.created_date_time[i]))
                     day3.append (find_day (X.created_date_time[i]))
                     year3.append (find_year (X.created_date_time[i]))
                 i += 1
                 
        
        i += 1




# Plotting daily status of hubs
        
for file_name in file_names:
    try:
      path = file_name
      dataset = pd.read_csv ((path))
    except:
      continue
    
    X = dataset.iloc[:, :]

    timestamp1 = []
    timestamp2 = []
    timestamp3 = []    
    year1 = []
    year2 = []
    year3 = []
    month1 = []
    month2 = []
    month3 = []
    day1 = []
    day2 = []
    day3 = []
    room_temp = []
    heatsetpoint = []
    heating = []
    get_plot_data (X)
        
    data1 = pd.DataFrame (np.column_stack([year1, month1, day1, timestamp1, room_temp]), columns = ['year', 'month', 'day', 'time', 'room_temp'])
    data2 = pd.DataFrame (np.column_stack([year2, month2, day2, timestamp2, heatsetpoint]), columns = ['year', 'month', 'day', 'time', 'heatsetpoint'])
    data3 = pd.DataFrame (np.column_stack([year3, month3, day3, timestamp3, heating]), columns = ['year', 'month', 'day', 'time', 'heating'])

    
    pp = PdfPages ('../Figures/' + os.path.basename (path) + '.pdf')
    for year in range (2017, 2020):
        for month in range (1, 13):
            for day in range (1, 32):
            
                X1 = data1.loc[(data1['month'] == month) & (data1['day'] == day) & (data1['year'] == year)]['time']
                X2 = data2.loc[(data2['month'] == month) & (data2['day'] == day) & (data2['year'] == year)]['time']
                X3 = data3.loc[(data3['month'] == month) & (data3['day'] == day) & (data3['year'] == year)]['time']
                y1 = data1.loc[(data1['month'] == month) & (data1['day'] == day) & (data1['year'] == year)]['room_temp']
                y2 = data2.loc[(data2['month'] == month) & (data2['day'] == day) & (data2['year'] == year)]['heatsetpoint']
                y3 = data3.loc[(data3['month'] == month) & (data3['day'] == day) & (data3['year'] == year)]['heating']
                
                if len (X1) > 0 and len (X3) > 0:
                    fig = plt.figure (day)
                    ax = plt.subplot ()
                    plt.grid (linestyle=':', linewidth=1)
                    plt.scatter (X1, y1, color = 'black')
                    plt.scatter (X3, y3, color = 'blue')
                    plt.scatter (X2, y2, color = 'red')
                    plt.xlabel ('Time of day')
                    plt.ylabel ('Room Temparature')
                    ax.set_xticks ([0,5,10,15,20,25])
                    ax.set_xticklabels (['00:00', '05:00', '10:00', '15:00', '20:00', '00:00'])
                    plt.title (os.path.basename (path) + "\nDay/Month/Year:  " + str (day) + '/' + str (month) + '/' + str (year))
                    plt.legend ()
                    plt.show ()
                    
                    pp.savefig (fig)
                    plt.close (fig)
    pp.close ()
