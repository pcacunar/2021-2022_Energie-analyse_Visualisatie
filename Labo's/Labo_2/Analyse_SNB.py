# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn



dataG=pd.read_csv('Grid_SNB.csv')
dataG['Time']=pd.to_datetime(dataG.Time,unit='ms')
dataG=dataG.set_index('Time',drop=True)


# Verbruiksprofiel
dataG['Total'].plot()

# Dag verbruiksprofiel per dag



plt.scatter(dataG.index.hour,dataG['Total'])
plt.xlabel('uur van de dag')
plt.ylabel('Vermogen [W]')
plt.title('Dagelijkse verbruiksprofiel voor SNB')



#Gemiddelde profiel
dataGh=pd.DataFrame()

dataGh['gemiddelde']=dataG.groupby(dataG.index.hour).mean()['Total']
dataGh['afwijking']=dataG.groupby(dataG.index.hour).std()['Total']

fig,ax=plt.subplots()
ax.plot(dataGh.index, dataGh.gemiddelde)
ax.fill_between(dataGh.index, dataGh.gemiddelde - dataGh.afwijking, dataGh.gemiddelde + dataGh.afwijking, alpha=0.35)
ax.set_ylabel('Vermogen W')
ax.axis([0,23,-400000,400000])

# Duurcurve

sort=np.sort(dataG['Total'])[::-1]
exceedence = np.arange(1.,len(sort)+1) / len(sort)

plt.plot(exceedence*100, sort)
plt.fill_between(exceedence*100, sort)
plt.xlabel("Gebeurtenis [%]")
plt.ylabel("Vermogen W")
plt.axis([0,100,sort.min(),sort.max()])
plt.show()


# Kansdichtheidsfunctie

dataG['Total'].plot.density()
plt.xlabel('Vermogen W')

# genormaliseerde verbruik

dataG['Verbruik']=dataG['Total']+dataZ['Total']
dataG['Verbruik'].plot(label='Verbruik')
dataG['Total'].plot(label='Werkelijke afn/inj')
dataZ['Total'].plot(label='Totaal productie')

dataG['Verbruik+']=dataG[(dataG['Verbruik']>0)]['Verbruik']
dataG['Verbruik+']=dataG['Verbruik+'].fillna(0)

dataG['Verbruik-']=dataG[(dataG['Verbruik']<0)]['Verbruik']
dataG['Verbruik-']=dataG['Verbruik-'].fillna(0)


genor=dataG['Verbruik+'].sum()*.25/1800


# Eigen/Afname/Injectie/Zc/Zv

dataG['Prod']=dataZ[(dataZ['Total']>=0)]['Total']
dataG['Prod']=dataG['Prod'].fillna(0)
dataG['Eig']=dataG[(dataG['Prod']>dataG['Verbruik+'])]['Verbruik+']
dataG['Eig']=dataG['Eig'].fillna(dataG['Prod'])

dataG['zv']=dataG['Eig']/dataG['Verbruik+']
dataG['zv']=dataG['zv'].fillna(0)
dataG['zv'].groupby(dataG.index.month).mean().plot.bar()
plt.title('Zelf-voorziening')
plt.xlabel('maand')


dataG['zc']=dataG['Eig']/dataG['Prod']
dataG['zc']=dataG['zc'].fillna(0)
dataG['zc'].groupby(dataG.index.month).mean().plot.bar()
plt.title('Zelf-consumptie')
plt.xlabel('maand')



# Fouriertransform
dataG['Total']=dataG.Total/dataG.Total.max()
t = np.arange(35041)
sp = np.fft.fft(dataG.Total)
sp=sp[range(int(len(dataG.Total)/2))]
# freq = np.arange(int(np.size(dataG.Total)/2))/(np.size(dataG.Total)/35041)#np.fft.fftfreq(365,1/35040)
freq=np.fft.fftfreq(t.size,1/35041)[:t.size//2]
plt.plot(freq, abs(sp))
plt.xlabel('frequentie [1/j]')
plt.ylabel('spectrale densiteit [a.u.]')
plt.axis([-5,400,0,max(abs(sp))*1.1])


# *************************************Productieprofiel

dataZ=pd.read_csv('Solar_SNB.csv')
dataZ['Time']=pd.to_datetime(dataZ.Time,unit='ms')
dataZ=dataZ.set_index('Time',drop=True)

dataZ['Total'].plot()
plt.ylabel('Vermogen W')



#Gemiddelde profiel
dataSh=pd.DataFrame()

dataSh['gemiddelde']=dataZ.groupby(dataZ.index.hour).mean()['Total']
dataSh['afwijking']=dataZ.groupby(dataZ.index.hour).std()['Total']

fig,ax=plt.subplots()
ax.plot(dataSh.index, dataSh.gemiddelde)
ax.fill_between(dataSh.index, dataSh.gemiddelde - dataSh.afwijking, dataSh.gemiddelde + dataSh.afwijking, alpha=0.35)
ax.set_ylabel('Vermogen W')
ax.axis([0,23,-6000,100000])

# Duurcurve voor productie

sort=np.sort(dataZ['Total'])[::-1]
exceedence = np.arange(1.,len(sort)+1) / len(sort)

plt.plot(exceedence*100, sort)
plt.fill_between(exceedence*100, sort)
plt.xlabel("Gebeurtenis [%]")
plt.ylabel("Vermogen W")
plt.axis([0,100,sort.min(),sort.max()])
plt.title('Totale zonne productie')
plt.show()


# Kansdichtheidsfunctie voor productie

dataZ['Total'].plot.density()
plt.xlabel('Vermogen W')

# Fouriertransform
dataZ['Total']=dataZ['Total']/dataZ['Total'].max()
t = np.arange(35041)
sp = np.fft.fft(dataZ['Total'])
sp=sp[range(int(len(dataZ['Total'])/2))]
# freq = np.arange(int(np.size(dataG.Total)/2))/(np.size(dataG.Total)/35041)#np.fft.fftfreq(365,1/35040)
freq=np.fft.fftfreq(t.size,1/35041)[:t.size//2]
plt.plot(freq, abs(sp))
plt.xlabel('frequentie [1/j]')
plt.ylabel('spectrale densiteit [a.u.]')
plt.axis([-5,400,0,max(abs(sp))*1.1])