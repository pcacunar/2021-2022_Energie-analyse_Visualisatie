# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:00:28 2021

@author: paula.acuna.roncanc1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data inlezen
data=pd.read_csv('DataSet_CampusKortrijk_Sep1_Oct1.csv')
units=data.iloc[0,:]
units=units.drop(['_start','_stop','_time'])
data=data.iloc[1:,:]
data=data.set_index('_time',drop=True)
data.index=pd.to_datetime(data.index)
data=data.drop(['_start','_stop'],axis=1)


# Data pre-processing
data=data.fillna(0)
data=data.astype(float)



def naarkW(data,units):
    datakW=pd.DataFrame()
    for i in range(data.iloc[1,:].size):
        if units[i]=='kW':
            datakW=pd.concat([datakW,data.iloc[:,i]],axis=1)
        else:
            datakW=pd.concat([datakW,data.iloc[:,i]/1000],axis=1)
    return datakW

datakW=naarkW(data,units)

datakW.index=pd.to_datetime(datakW.index)
# datakW=datakW[(datakW.index.month==9) & (datakW.index.day==12)]
    

def maandver(data,gebouw):
    ener=data[gebouw].sum()*.25
    return ener

def duurcurve(data,feature,ax):
    sort=np.sort(data[feature])[::-1]
    exceedence = np.arange(1.,len(sort)+1) / len(sort)
    
    ax.plot(exceedence*100, sort)
    ax.fill_between(exceedence*100, sort)
    ax.set_xlabel("Gebeurtenis [%]")
    ax.set_ylabel("Vermogen kW")
    ax.set(xlim=(0,100),ylim=(sort.min(),sort.max()))
    return

def kans(data,gebouw,ax):
    data[gebouw].plot.density(ax=ax)
    ax.set_xlabel('Vermogen kW')
    return

def fourierMaan(data,gebouw,ax):
    data[gebouw]=data[gebouw]/data[gebouw].max()
    t = np.arange(2880)
    sp = np.fft.fft(data[gebouw])
    sp=sp[range(int(len(data[gebouw])/2))]
    # freq = np.arange(int(np.size(dataG.Total)/2))/(np.size(dataG.Total)/35041)#np.fft.fftfreq(365,1/35040)
    freq=np.fft.fftfreq(t.size,1/2880)[:t.size//2]
    ax.plot(freq, abs(sp))
    ax.set_xlabel('frequentie [1/j]')
    ax.set_ylabel('spectrale densiteit [a.u.]')
    ax.set(xlim=(-5,300),ylim=(0,max(abs(sp))*1.1))
    return



# OBEE

datakW['OBEE_Afname']=datakW[['G_OB_VermogenL1_kW','G_OB_VermogenL2_kW','G_OB_VermogenL3_kW']].sum(axis=1)



ene_geb=[]

for i in range(datakW.columns.size):
    ene_geb.append(round(maandver(datakW,datakW.columns[i]),1))
    print('Maandelijksverbruik van '+datakW.columns[i]+' is '+str(round(maandver(datakW,datakW.columns[i]),1))+' kWh voor september')
    plt.figure(i+1)
    fig, axs = plt.subplots(2,3)
    fig.suptitle(datakW.columns[i])
    datakW[datakW.columns[i]].plot(ax=axs[0,0])
    duurcurve(datakW,datakW.columns[i],axs[0,1])
    kans(datakW,datakW.columns[i],axs[0,2])
    fourierMaan(datakW,datakW.columns[i],axs[1,0])
    sns.boxplot(data=datakW,x=datakW.index.weekday,y=datakW.columns[i],ax=axs[1,1])
    sns.boxplot(data=datakW,x=datakW.index.hour,y=datakW.columns[i],ax=axs[1,2])
    
ene_geb=np.array(ene_geb)
ene_geb=pd.DataFrame(ene_geb,index=datakW.columns)

ene_geb=ene_geb.sort_values(ene_geb.columns[0],ascending=0)
ene_geb.plot.bar()

#*********************** Gebouwen met HEB *************************************************************

def zc_zv(data,grid,prod,leyenda):
    dataHEB=pd.DataFrame()
    if data[prod].mean()>0:
        dataHEB[leyenda+'_verbr']=data[grid]+data[prod]
    else:
        dataHEB[leyenda+'_verbr']=data[grid]-data[prod]
    dataHEB[leyenda+'_eigen']=np.zeros(dataHEB[leyenda+'_verbr'].size)
    for i in range(dataHEB.iloc[:,0].size):
        if dataHEB.iloc[i,0]>=data[prod].iloc[i]:
            dataHEB[leyenda+'_eigen'].iloc[i]=data[prod].iloc[i]
        else:
            dataHEB[leyenda+'_eigen'].iloc[i]=dataHEB[leyenda+'_verbr'].iloc[i]
    zc=dataHEB[leyenda+'_eigen'].sum()/abs(data[prod].sum())
    zv=dataHEB[leyenda+'_eigen'].sum()/dataHEB[leyenda+'_verbr'].sum()
    
    print('Voor de '+leyenda+' de zelf-voorziening is '+str(round(zv,1))+' en de zelf-consumptie is '+str(round(zc,1)))
    return 

# De reactor

zc_zv(datakW,'De_Reactor_Afname_W','De_Reactor_Productie_W','DeReactor')

# Gang 100-200

zc_zv(datakW,'G_A_100-200_Vermogen_W','PV_IoT_Vermogen_W','Gang_100-200')

# Gang 300-500


zc_zv(datakW,'GebA_300-500','Lemcko_Pvproductie_kW','Gang_300-500')

# Penta
datakW['Penta_opbrengst']=datakW[['Penta_Pvproductie_kW','Penta_WKKproductie_kW']].sum(axis=1)
zc_zv(datakW,'Penta_Afname_kW','Penta_opbrengst','Penta')

# AD+IPD

zc_zv(datakW,'AD_IDC_afname_W','AD_IDC_Pvproductie_W','AD+ID')






    