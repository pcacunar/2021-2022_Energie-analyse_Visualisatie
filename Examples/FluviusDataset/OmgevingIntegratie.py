# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:02:11 2021

@author: paula.acuna.roncanc1
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sn


dataW=pd.read_csv('POWER_nasa_dataset/POWER_Point_Hourly_20180601_20210630_050d8315N_003d2317E_LST.csv',skiprows=17)
dataW=dataW.rename(columns={'YEAR':'year','MO':'month','DY':'day','HR':'hour'})
dataW['indx']=pd.to_datetime(dataW[['year','month','day','hour']])
dataW=dataW.set_index('indx',drop=True)

dataW=dataW.resample('15T').ffill()

dataW=dataW[(dataW.index>datetime.datetime(2018,6,16,23,45)) & (dataW.index<=datetime.datetime(2021,6,16,23,45))]

dataA41=dataA41[(dataA41.index>datetime.datetime(2018,6,16,23,45)) & (dataA41.index<=datetime.datetime(2021,6,16,23,45))]

dataMaster=pd.concat([dataW,dataA41],axis=1)

plt.scatter(dataMaster.T2M,dataMaster.Vermogen_kW)
plt.xlabel('temperatuur')
plt.ylabel('vermogen')

dataMaster=dataMaster.drop(['year','month','day','hour'],axis=1)

sn.pairplot(dataMaster,size=0.8)
