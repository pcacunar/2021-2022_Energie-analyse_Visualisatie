# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:27:38 2021

@author: paula.acuna.roncanc1
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# Data inlezen

data=pd.read_csv('WindFarms_complementary_data.csv') #,skiprows=1
data.Time_UTC=pd.to_datetime(data.Time_UTC,format='%d-%m-%y %H:%M')
data=data.set_index('Time_UTC',drop=True)


dict_wf={}
for _ in range(data.Wind_Farm.unique().size):
    name='dataWF_'+str(_)
    dict_wf[name]=data[data.Wind_Farm==data.Wind_Farm.unique()[_]]
    
    
dictWF1_t={}
for _ in range(dict_wf['dataWF_1'].Wind_Turbine.unique().size):
    name='dataWF1_T_'+str(_)
    dictWF1_t[name]=dict_wf['dataWF_1'][dict_wf['dataWF_1'].Wind_Turbine==dict_wf['dataWF_1'].Wind_Turbine.unique()[_]]


# dictWF1_t['dataWF1_T_0']['Average_power_output_MW'].plot()
# dictWF2_t['dataWF2_T_0']['Average_power_output_MW'].plot()
# dictWF3_t['dataWF3_T_0']['Average_power_output_MW'].plot()
# dictWF4_t['dataWF4_T_0']['Average_power_output_MW'].plot()

WF1_t0=dictWF1_t['dataWF1_T_0']
WF1_t0=WF1_t0.fillna(0)

seasonal_decompose(WF1_t0['Average_power_output_MW'],model='additive').plot()


WF1_t0['Maand']=WF1_t0.index.month
WF1_t0['UvD']=WF1_t0.index.hour

sns.boxplot(data=WF1_t0,x='UvD',y='Average_power_output_MW')

sns.boxplot(data=WF1_t0,x='Maand',y='Average_power_output_MW')


