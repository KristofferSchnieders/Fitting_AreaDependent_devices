# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:52:19 2024

@author: schnieders
"""

import numpy as np 
import pandas as bearcats 
import os 
import sys
import collections  
import matplotlib.pyplot as plt
import tqdm 
import scipy

from matplotlib import cm
from measurement_control.files import TextFile

from scipy.optimize import curve_fit

plt.rc('font', weight='bold')
props = dict(boxstyle='round', facecolor='white', alpha=1)


# customize ticks 
plt.rc("font",size=18, weight="bold")

# customize labels 
font_labels = {'labelsize' : 15,
        'labelweight' : 'bold'}
plt.rc('axes', **font_labels)

# customize title 
font_title = {'titlesize' : 18,
        'titleweight' : 'bold',
        'titlepad' : 20}
plt.rc('axes', **font_title)             

# Customize font in math environment
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)  


plt.rc('font', weight='bold')
props = dict(boxstyle='round', facecolor='white', alpha=1)

dir_data = r"\\iff1454\transfer\user\Schnieders\M-CNN\PCMO\1nmAlO_standard_annealed"
        

# Methode to find all files in subfolders of directory.
# This methode is used to find all rawdata in subfolders of folder handed in. 
# Be cautions that folder you handed over is the right one. 
def find_all_files_in_dir(dir_data: str):
    '''
    Methode to find all files in folder

    Parameters
    ----------
    dirertory : string
        Main folder in which raw data is stored in. 
        
    Returns
    -------
    file_list : list
        list of the directories of all files in folder itself and of subfolders. 

    '''
    
    list_of_subpaths = os.listdir(dir_data)
    files = list()
    
    # Search iteravely in all subfolders
    for sub_path in list_of_subpaths:
        
        current_path = os.path.join(dir_data, sub_path)
        
        # iterate procedure, if directory is folder. Otherwise store in files  
        if os.path.isdir(current_path):
            files = files + find_all_files_in_dir(current_path)
        else:
            files.append(current_path)
                
    return files
#%%
files = np.array(find_all_files_in_dir(dir_data))
files = np.array([file for file in files if ".txt" in file])
devices = list(collections.Counter([file.split("\\")[np.where(["100" in file_part for file_part in file.split("\\")])[0][0]] for file in files]))
for device in devices: 
    files_dev = files[[device in file for file in files]]
    for file in tqdm.tqdm(files_dev, desc="Data of device " + device): 
        
        if "presweep" in file: 
            action = "sweep"
            max_sweep = 0
        elif "define_res_sweep" in file: 
            action =  "set_device"
        elif "current_pulse" in file:
            action = "pulse"
        
        data_pd = TextFile(file).read().data
        I, V = data_pd["i"].to_numpy(), data_pd["v"].to_numpy()
        try:
            if "presweep" in file: 
                action = "sweep"
                max_sweep = 0
            elif "define_res_sweep" in file: 
                action =  "set_device"
                zero_crossing = np.where(np.diff(np.sign(I))!=0)[0]
                I_dummy = I[zero_crossing[-1]:]
                max_sweep = np.sign(I_dummy[np.argmax(abs(I_dummy))])*np.max(abs(I_dummy))
            elif "current_pulse" in file:
                action = "pulse"
        except:
            pass
        time_meas = os.path.getmtime(file)
        dictionary_of_data = {"file": file, 
                              "action":action,
                              "I": I, 
                              "sweep_state": max_sweep,
                              "V": V, 
                              "device":device,
                              "t": data_pd["t"].to_numpy(),
                              "time": time_meas
                              }
    
        if "df" not in vars(): 
            df = bearcats.DataFrame([dictionary_of_data])
        else:
            df =  bearcats.concat([df, bearcats.DataFrame([dictionary_of_data])], ignore_index = True)
df = df.sort_values(by=["time"],ignore_index=True)
#%%
cmap = cm.rainbow

for device in devices: 
    V = np.concatenate(df.V[np.logical_and(df.device==device, df.action=="sweep")].to_numpy(),axis=0)
    I = np.concatenate(df.I[np.logical_and(df.device==device, df.action=="sweep")].to_numpy(),axis=0)
    
    zero_crossing = np.where(np.diff(np.sign(V))!=0)[0]
    nr_cycle = len(zero_crossing[::2])-1
    for cycle in range(nr_cycle):
        cval = cmap((cycle+0.5)/nr_cycle)
        plt.plot(V[zero_crossing[cycle]: zero_crossing[cycle+1]], 
                 I[zero_crossing[cycle]: zero_crossing[cycle+1]]*1e3, 
                 color=cval, zorder=2)
    
    plt.xlabel("V / V")
    plt.ylabel("I / $\mu$A")
    plt.title(device)
    plt.show()
    
#%%
plt.scatter(df[df.device==device].sweep_state, [np.sign(V[int(len(V)/2)])*max(abs(V)) for V in df[df.device==device].V])
plt.xlabel("Presweep")
plt.ylabel("Current pulse")
plt.title(device)
plt.show()
#%%
nr_trace =5
device = "L10_100_100"
index = df[df.device==device].index[nr_trace]
V, I, t = df.V[index], df.I[index]*1e6, df.t[index]
plt.plot(I)
V, I, t = V[abs(I)>1e1][1:-1], I[abs(I)>1e1][1:-1], t[abs(I)>1e1][1:-1]
t = t-min(t)
plt.plot(t, I)
plt.xlabel("time/ s")
plt.ylabel("I / $\mu$A")
plt.title("Example")
plt.show()

plt.plot(t,V)
plt.xlabel("time/ s")
plt.ylabel("V / V")
plt.title("Example")
plt.show()

# Set artificial points for input current of 0 and perhaps add exception for this case.
'HP paper' 
def func(t, a, b, c, d, a1, b1, d1):

    return a * np.exp(b * t**d) - a1 * np.exp(-b1 * t**d1) + c

tfit = np.linspace(min(t), max(t), len(t))
popt, pcov = curve_fit(func, tfit, V,method="trf")
slope, intercept, r, p, se = scipy.stats.linregress(V, func(tfit, *popt))
plt.plot(t, V)
plt.plot(tfit, func(tfit, *popt))
plt.xlabel("time/ s")
plt.ylabel("V / V")
plt.title("Fit")
plt.legend()
plt.show()
#%%