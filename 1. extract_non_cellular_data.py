# Extracting non cellular data
# Outputs non - cellular data in machine_learning/EthernetData folder.

# Importing Libraries
import pandas as pd
import numpy as np
import glob
import os


# Importing the Dataset
file_names = glob.glob ('./Hub_query/Hub*')


# Extract hub data with network type ethernet
import json
def extract_ethernet_data (X):
    thermo = pd.DataFrame (columns = ['created_date_time', 'content', 'type'])            
    i = 0
    while i < len (X):
        content = json.loads (X.content[i])
        if 'status' in content and "network_type" in content["status"]:
            if content["status"]["network_type"] in ['ethernet']:
                i += 1
                while i < len (X):
                    content = json.loads (X.content[i])
                    if 'status' in content and "network_type" in content["status"]:
                        if content["status"]["network_type"] in ['cellular']:
                            break
                    thermo = thermo.append (X.loc[i], ignore_index = True)
                    i += 1    
        i += 1
        
    thermo = thermo.loc[thermo['type'] == 'thermostat'].reset_index (drop = True)
    thermo = thermo[['created_date_time', 'content']]
    
    if len (thermo) > 1000:
        path = './'
        if not os.path.exists (path):
            os.makedirs (path)
        thermo.to_csv (path + 'Hub_' + X.hub_id[0] + '.csv', index = False)
        
    
curr = 0
# Run extract ethernet data for all hubs
for file in file_names:
    print ('Processing = ' + str (curr))
    curr += 1
    try:    
        dataset = pd.read_csv (file)
    except:
        continue
    X = dataset.iloc[:,:]
    X = X[['hub_id', 'created_date_time', 'content', 'type']]
    extract_ethernet_data(X)
    

