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


# Productieprofiel

dataZ=pd.read_csv('Solar_SNB.csv')
dataZ['Time']=pd.to_datetime(dataZ.Time,unit='ms')
dataZ=dataZ.set_index('Time',drop=True)

dataZ['Total'].plot()
plt.ylabel('Vermogen W')

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