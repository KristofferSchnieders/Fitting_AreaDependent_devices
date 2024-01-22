# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:52:19 2024

@author: schnieders
"""

import matplotlib.pyplot as plt

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

