# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names=['From','To','meterID','seq','Bx','23','E12-E17','kWx','A_C_I','Plaats']
for i in np.arange(0,96):
    names.append('kwartier_'+str(i))
    
data=pd.read_csv('Data/AMR_REPORTING_EXPORT_1juli21_dataset1.csv',delimiter=';',index_col=False)
data=data.iloc[:,:-5]
data.columns=names

data.From=pd.to_datetime(data['From'],format='%d%m%Y %H:%M')
data=data.sort_values('From')

inter=data.A_C_I.unique()

dataAct_a=data.loc[data['A_C_I']=='A+']

dataInd_a=data.loc[data['A_C_I']=='I+']

dataCap_a=data.loc[data['A_C_I']=='C-']

dataX=data.loc[data['A_C_I'].isna()]



def transfoE(data,startDatum):
    dataA=[]
    for i in np.arange(0,data.iloc[:,0].size):
        temp=data.iloc[i,10:].transpose().values
        for j in np.arange(0,temp.size):
            if not(pd.isna(temp[j])):
                dataA.append(temp[j])
    dataA=np.array(dataA)
    datee=pd.date_range(start=startDatum,periods=dataA.size,freq='15T')
    dataA=pd.DataFrame({'Date':datee,'Energie_kWh':dataA})
    dataA=dataA.set_index('Date',drop=True)
    dataA.Energie_kWh.plot(label='afname')
    plt.ylabel('Energie verbruik per kwartier [kWh]')
    return dataA

# Elektriciteit
metID='541449500001660041'
tempo=dataAct_a[dataAct_a['meterID']==metID]
dataA=transfoE(dataAct_a[dataAct_a['meterID']==metID],dataAct_a[dataAct_a['meterID']==metID].From.iloc[0])
dataA.to_csv('ActieveAfname_ID'+metID+'.csv')

tempo=dataInd_a[dataInd_a['meterID']==metID]
dataI=transfoE(dataInd_a[dataInd_a['meterID']==metID],dataInd_a[dataInd_a['meterID']==metID].From.iloc[0])
dataI.to_csv('InductieveAfname_ID'+metID+'.csv')

tempo=dataCap_a[dataCap_a['meterID']==metID]
dataI=transfoE(dataCap_a[dataCap_a['meterID']==metID],dataCap_a[dataCap_a['meterID']==metID].From.iloc[0])
dataI.to_csv('CapacitieveAfname_ID'+metID+'.csv')


def transfoG(data,startDatum,label):
    dataA=[]
    for i in np.arange(0,data.iloc[:,0].size):
        temp=data.iloc[i,10:].transpose().values
        for j in np.arange(0,temp.size):
            if not(pd.isna(temp[j])):
                dataA.append(temp[j])
    dataA=np.array(dataA)
    datee=pd.date_range(start=startDatum,periods=dataA.size,freq='H')
    dataA=pd.DataFrame({'Date':datee,label:dataA})
    dataA=dataA.set_index('Date',drop=True)
    dataA[label].plot(label='afname')
    plt.ylabel(label)
    return dataA

# Gas

metID='541448860012075359'
tempo=dataAct_a[dataAct_a['meterID']==metID]
dataA=transfoG(dataAct_a[dataAct_a['meterID']==metID],dataAct_a[dataAct_a['meterID']==metID].From.iloc[0],'Energie_kWh')
dataA.to_csv('Gas_ActieveAfname_ID'+metID+'.csv')

metID='SUB(541448860012075359)'
tempo=dataAct_a[dataAct_a['meterID']==metID]
dataA_MTQ=transfoG(tempo[tempo['kWx']=='MTQ'],tempo[tempo['kWx']=='MTQ'].From.iloc[0],'m3')
dataA_MTQ.to_csv('Gas_verbruik m3_ID'+metID+'.csv')

metID='SUB(541448860012075359)'
tempo=dataAct_a[dataAct_a['meterID']==metID]
dataA_D90=transfoG(tempo[tempo['kWx']=='D90'],tempo[tempo['kWx']=='D90'].From.iloc[0],'m3N')
dataA_D90.to_csv('Gas_verbruik m3N_ID'+metID+'.csv')

dataGas=dataA.Energie_kWh/dataA_D90.m3N
dataGas.plot()
plt.ylabel('kWh/m3N')
dataGas.to_csv('Gas_kWh_per_m3N_ID'+metID+'.csv')
