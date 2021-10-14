# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
import seaborn as sn
import streamlit as st
from matplotlib.figure import Figure

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Analyse van FLUVIUS dataset')

st.header('Data inlezen en omzetten')

names=['From','To','meterID','seq','Bx','23','E12-E17','kWx','A_C_I','Plaats']
for i in np.arange(0,96):
    names.append('kwartier_'+str(i))
      
    
data=pd.read_csv('AMR_REPORTING_EXPORT_depurated.csv',names=names,delimiter=';',index_col=False)
data.From=pd.to_datetime(data['From'],format='%d%m%Y %H:%M')
data=data.sort_values('From')

data['kwartier_6']=data.kwartier_6.astype(float)
data['kwartier_18']=data.kwartier_6.astype(float)
data['kwartier_21']=data.kwartier_6.astype(float)
data['kwartier_25']=data.kwartier_6.astype(float)

inter=data.A_C_I.unique()

dataAct_a=data.loc[data['A_C_I']=='A+']

dataInd_a=data.loc[data['A_C_I']=='I+']

dataCap_a=data.loc[data['A_C_I']=='C-']

dataX=data.loc[data['A_C_I'].isna()]

st.code('''
        names=['From','To','meterID','seq','Bx','23','E12-E17','kWx','A_C_I','Plaats']
for i in np.arange(0,96):
    names.append('kwartier_'+str(i))
      
    
data=pd.read_csv('AMR_REPORTING_EXPORT_depurated.csv',names=names,delimiter=';',index_col=False)
data.From=pd.to_datetime(data['From'],format='%d%m%Y %H:%M')
data=data.sort_values('From')

data['kwartier_6']=data.kwartier_6.astype(float)
data['kwartier_18']=data.kwartier_6.astype(float)
data['kwartier_21']=data.kwartier_6.astype(float)
data['kwartier_25']=data.kwartier_6.astype(float)

inter=data.A_C_I.unique()

dataAct_a=data.loc[data['A_C_I']=='A+']

dataInd_a=data.loc[data['A_C_I']=='I+']

dataCap_a=data.loc[data['A_C_I']=='C-']

dataX=data.loc[data['A_C_I'].isna()]''')


st.subheader('Omzettingsfunctie')


def transfoE(data,startDatum):

    if data.kWx.unique()[0] =='KWH':
        dataA=[]
        for i in np.arange(0,data.iloc[:,0].size):
            temp=data.iloc[i,10:34].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataA=np.array(dataA)
        lab='Energie_kWh'
        uni='kWh'
        datee=pd.date_range(start=startDatum,periods=dataA.size,freq='1H')
        dataA=pd.DataFrame({'Date':datee,lab:dataA})
        dataA=dataA.set_index('Date',drop=True)

    elif data.kWx.unique()[0]=='KWT':
        dataA=[]
        for i in np.arange(0,data.iloc[:,0].size):
            temp=data.iloc[i,10:].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataA=np.array(dataA)
        lab='Vermogen_kW'
        uni='kW'
        datee=pd.date_range(start=startDatum,periods=dataA.size,freq='15T')
        dataA=pd.DataFrame({'Date':datee,lab:dataA})
        dataA=dataA.set_index('Date',drop=True)
        
    else: 
        dataMTQ=data[data.kWx=='MTQ']
        dataA=[]
        for i in np.arange(0,dataMTQ.iloc[:,0].size):
            temp=dataMTQ.iloc[i,10:34].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataAMTQ=np.array(dataA)
        lab1='Energie_m3'
        uni='m3'
        datee=pd.date_range(start=startDatum,periods=dataAMTQ.size,freq='1H')
        
        dataD90=data[data.kWx=='D90']
        dataA=[]
        for i in np.arange(0,dataD90.iloc[:,0].size):
            temp=dataD90.iloc[i,10:34].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataAD90=np.array(dataA)
        lab2='Energie_m3N'
        uni='m3N'
        datee=pd.date_range(start=startDatum,periods=dataAD90.size,freq='1H')
        dataA=pd.DataFrame({'Date':datee,lab1:dataAMTQ,lab2:dataAD90})
        dataA=dataA.set_index('Date',drop=True)


    
    # dataA.plot()
    st.line_chart(dataA)
    # plt.ylabel(lab+' per kwartier ['+uni+']')
    return dataA

st.code('''
        def transfoE(data,startDatum):

    if data.kWx.unique()[0] =='KWH':
        dataA=[]
        for i in np.arange(0,data.iloc[:,0].size):
            temp=data.iloc[i,10:34].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataA=np.array(dataA)
        lab='Energie_kWh'
        uni='kWh'
        datee=pd.date_range(start=startDatum,periods=dataA.size,freq='1H')
        dataA=pd.DataFrame({'Date':datee,lab:dataA})
        dataA=dataA.set_index('Date',drop=True)

    elif data.kWx.unique()[0]=='KWT':
        dataA=[]
        for i in np.arange(0,data.iloc[:,0].size):
            temp=data.iloc[i,10:].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataA=np.array(dataA)
        lab='Vermogen_kW'
        uni='kW'
        datee=pd.date_range(start=startDatum,periods=dataA.size,freq='15T')
        dataA=pd.DataFrame({'Date':datee,lab:dataA})
        dataA=dataA.set_index('Date',drop=True)
        
    else: 
        dataMTQ=data[data.kWx=='MTQ']
        dataA=[]
        for i in np.arange(0,dataMTQ.iloc[:,0].size):
            temp=dataMTQ.iloc[i,10:34].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataAMTQ=np.array(dataA)
        lab1='Energie_m3'
        uni='m3'
        datee=pd.date_range(start=startDatum,periods=dataAMTQ.size,freq='1H')
        
        dataD90=data[data.kWx=='D90']
        dataA=[]
        for i in np.arange(0,dataD90.iloc[:,0].size):
            temp=dataD90.iloc[i,10:34].transpose().values
            for j in np.arange(0,temp.size):
                if not(pd.isna(temp[j])):
                    dataA.append(temp[j])
        dataAD90=np.array(dataA)
        lab2='Energie_m3N'
        uni='m3N'
        datee=pd.date_range(start=startDatum,periods=dataAD90.size,freq='1H')
        dataA=pd.DataFrame({'Date':datee,lab1:dataAMTQ,lab2:dataAD90})
        dataA=dataA.set_index('Date',drop=True)


    
    dataA.plot()
    # plt.ylabel(lab+' per kwartier ['+uni+']')
    return dataA
        ''')

# Elektriciteit

st.subheader('Dataframes voor elke van de meters')

st.code('''
        meters=data.meterID.unique()
def datameter(meter):
    tempo=dataAct_a[(dataAct_a['meterID']==meter)] # & (dataAct_a['kWx']=='KWT')
    dataA=transfoE(tempo,tempo.From.iloc[0])
    return dataA

dataA41=datameter(meters[0])
dataA59=datameter(meters[1])
dataAsub59=datameter(meters[2])
plt.legend()
        ''')

meters=data.meterID.unique()
def datameter(meter):
    tempo=dataAct_a[(dataAct_a['meterID']==meter)] # & (dataAct_a['kWx']=='KWT')
    dataA=transfoE(tempo,tempo.From.iloc[0])
    return dataA

dataA41=datameter(meters[0])
dataA59=datameter(meters[1])
dataAsub59=datameter(meters[2])
plt.legend()


    

# Daarnaast is het ook belangrijk om te weten de resolutie van elke meter. 
dataA41.head(5)
# Kwartier resolutie

dataA59.head(5)
# Uur resolutie

dataAsub59.head(5)
# Uur resolutie



# Wij zullen uithalen de data voor elke van de meters en uitdrukkend hoeveel is de energie per gemeten periode

print('Het energieverbruik gemeten door de meter '+meters[0]+' is gelijk aan '+str(dataA41['Vermogen_kW'].sum()*.25/1000)+'MWh en werd gemeten tussen '+str(dataA41.index.min())+' en '+str(dataA41.index.max()))

print('Het energieverbruik gemeten door de meter '+meters[1]+' is gelijk aan '+str(dataA59['Energie_kWh'].sum()/1000)+'MWh en werd gemeten tussen '+str(dataA59.index.min())+' en '+str(dataA59.index.max()))

print('Het energieverbruik gemeten door de meter '+meters[2]+' is gelijk aan '+str(dataAsub59['Energie_m3'].sum())+'m3 en werd gemeten tussen '+str(dataAsub59.index.min())+' en '+str(dataAsub59.index.max()))

st.header('Energieverbruik per meter')
st.markdown('Het energieverbruik gemeten door de meter '+meters[0]+' is gelijk aan '+str(dataA41['Vermogen_kW'].sum()*.25/1000)+'MWh en werd gemeten tussen '+str(dataA41.index.min())+' en '+str(dataA41.index.max()))

st.markdown('Het energieverbruik gemeten door de meter '+meters[1]+' is gelijk aan '+str(dataA59['Energie_kWh'].sum()/1000)+'MWh en werd gemeten tussen '+str(dataA59.index.min())+' en '+str(dataA59.index.max()))

st.markdown('Het energieverbruik gemeten door de meter '+meters[2]+' is gelijk aan '+str(dataAsub59['Energie_m3'].sum())+'m3 en werd gemeten tussen '+str(dataAsub59.index.min())+' en '+str(dataAsub59.index.max()))
# Baseload van elke van de meters
st.header('Baseload per meter')

st.subheader('Meter xx59')

st.write(sn.boxplot(data=dataA41,y=dataA41.columns[0]))
st.pyplot()

# dataA41.boxplot()


dataA41.groupby(dataA41.index.time).mean().plot()

st.write(dataA41[dataA41.index.date==datetime.date(2019,2,20)].plot())
st.pyplot()

st.line_chart(dataA41)

st.markdown('Wij zien dat voor de verbruiksprofiel van elektriciteit, de baseline is rond de eerste 25ste percentil - 180 kW')
st.subheader('Meter xx59')
# dataA59.boxplot()
st.write(sn.boxplot(data=dataA59,y=dataA59.columns[0]))
st.pyplot()

st.markdown('Naast de boxplot voor de totale tijd, is het belangrijk om de belastingsprofiel te kijken en nagaan als seizoensverbonden gedraag is')
dataA59['seizoen']=((dataA59.index.month<=3))*1 + ((dataA59.index.month>3) & (dataA59.index.month<=6))*2 + ((dataA59.index.month>6) & (dataA59.index.month<=9))*3 |+ ((dataA59.index.month>9) & (dataA59.index.month<=12))*4

st.write(sn.boxplot(data=dataA59,x='seizoen',y=dataA59.columns[0]))
st.pyplot()
plt.title('Seizoen met 1 gelijk aan winter')


# Lineplot
# plt.figure(2)
fig=Figure()
ax=fig.subplots()
winter=dataA59[dataA59.index.date==datetime.date(2020,2,20)]['Energie_kWh']
winter.index=winter.index.strftime('%H:%M')
winter.plot(label='winter',ax=ax)
lente=dataA59[dataA59.index.date==datetime.date(2020,4,20)]['Energie_kWh']
lente.index=lente.index.strftime('%H:%M')
lente.plot(label='lente',ax=ax)
zommer=dataA59[dataA59.index.date==datetime.date(2020,7,20)]['Energie_kWh']
zommer.index=zommer.index.strftime('%H:%M')
zommer.plot(label='zommer',ax=ax)
herfst=dataA59[dataA59.index.date==datetime.date(2020,10,20)]['Energie_kWh']
herfst.index=herfst.index.strftime('%H:%M')
herfst.plot(label='herfst',ax=ax)
ax.legend()
st.pyplot(fig)
# plt.legend()
st.markdown('Niet altijd de baseline is de minimum van de boxplot')

st.header('Piekvermogen voor de laatste 12 maanden')
st.markdown('De laatste gemeten dag is '+str(dataA41.index.max())+' dus, wij zullen nemen een jaar achteraf sinds dat moment nemen')

lastjaar=dataA41[(dataA41.index>(dataA41.index.max()+datetime.timedelta(days=-365)))]
lastjaarpiek=lastjaar.groupby(lastjaar.index.month).max()
st.write(lastjaarpiek.plot.bar())
plt.xlabel('Maand')
st.pyplot()

st.header('Correlatie tussen verschillende meters')
st.markdown('''OM de correlatie tussen de verschillende meters te kunnen bepalen, is het nodig dat de resolutie van alle meters gelijkaardig is.
            Wij zullen de resolutie van de elektrische meter ook naar uur brengen''')


dataA41H=dataA41.resample('1H').mean()

dataA41H=dataA41H[(dataA41H.index>=datetime.datetime(2018,6,17,19,0)) & (dataA41H.index<=datetime.datetime(2021,6,17,19,0))]

dataA59=dataA59[(dataA59.index>=datetime.datetime(2018,6,17,19,0)) & (dataA59.index<=datetime.datetime(2021,6,17,19,0))]

dataAsub59=dataAsub59[(dataAsub59.index>=datetime.datetime(2018,6,17,19,0)) & (dataAsub59.index<=datetime.datetime(2021,6,17,19,0))]
dataMaster=pd.concat([dataA41H,dataA59,dataAsub59],axis=1)
dataMaster['Elek_Energie_kWh']=dataMaster['Vermogen_kW']*.25
dataMaster=dataMaster.drop('Vermogen_kW',axis=1)

st.code('''
        dataA41H=dataA41.resample('1H').mean()

dataA41H=dataA41H[(dataA41H.index>=datetime.datetime(2018,6,17,19,0)) & (dataA41H.index<=datetime.datetime(2021,6,17,19,0))]

dataA59=dataA59[(dataA59.index>=datetime.datetime(2018,6,17,19,0)) & (dataA59.index<=datetime.datetime(2021,6,17,19,0))]

dataAsub59=dataAsub59[(dataAsub59.index>=datetime.datetime(2018,6,17,19,0)) & (dataAsub59.index<=datetime.datetime(2021,6,17,19,0))]
dataMaster=pd.concat([dataA41H,dataA59,dataAsub59],axis=1)
dataMaster['Elek_Energie_kWh']=dataMaster['Vermogen_kW']*.25
dataMaster=dataMaster.drop('Vermogen_kW',axis=1)

        ''')

st.dataframe(dataMaster)

# Correlatie tussen verschillende meters

st.write(sn.heatmap(dataMaster.corr(),cmap='Wistia',vmin=-1,vmax=1,annot=True))
st.pyplot()

dataMaster=dataMaster.drop('seizoen',axis=1)


st.header('Verandering jaar a jaar per meter')

st.write(dataMaster.groupby(dataMaster.index.year).sum().plot.bar())
plt.title('Let op, de jaren 2018 en 2021 zijn niet volledig')
st.pyplot()


st.header('Correlatie met de omgeving')

dataW=pd.read_csv('POWER_Hourly_20180601_20210630.csv',skiprows=17)
dataW=dataW.rename(columns={'YEAR':'year','MO':'month','DY':'day','HR':'hour'})
dataW['indx']=pd.to_datetime(dataW[['year','month','day','hour']])
dataW=dataW.set_index('indx',drop=True)
dataW=dataW[(dataW.index>=datetime.datetime(2018,6,17,19,0)) & (dataW.index<=datetime.datetime(2021,6,17,19,0))]

dataW=dataW.drop(['year','month','day','hour'],axis=1)

dataMaster=pd.concat([dataMaster,dataW],axis=1)

st.code('''
        dataW=pd.read_csv('POWER_Hourly_20180601_20210630.csv',skiprows=17)
dataW=dataW.rename(columns={'YEAR':'year','MO':'month','DY':'day','HR':'hour'})
dataW['indx']=pd.to_datetime(dataW[['year','month','day','hour']])
dataW=dataW.set_index('indx',drop=True)
dataW=dataW[(dataW.index>=datetime.datetime(2018,6,17,19,0)) & (dataW.index<=datetime.datetime(2021,6,17,19,0))]

dataW=dataW.drop(['year','month','day','hour'],axis=1)

dataMaster=pd.concat([dataMaster,dataW],axis=1)
        ''')

# Heatmap van de correlatie met de omgevingsvariabelen
fig, ax = plt.subplots(figsize=(10,10)) 
st.write(sn.heatmap(dataMaster.corr(),cmap='Wistia',vmin=-1,vmax=1,annot=True,ax=ax))
st.pyplot()

# Pairplot van de volledige dataset

st.write(sn.pairplot(dataMaster,size=1.2))
st.pyplot()

st.header('Facturatie')

# Elektriciteit



piekMei21=dataA41[(dataA41.index.month==5)]['Vermogen_kW'].max()
print('Piekvermogen te beschouwen in de berekening is '+str(piekMei21)+'kW')
verbMei21=dataA41[(dataA41.index.month==5)]['Vermogen_kW'].sum()*.25/1000
print('Verbruik te beschouwen in de berekening is '+str(verbMei21)+'MWh')

st.code('''
        
piekMei21=dataA41[(dataA41.index.month==5)]['Vermogen_kW'].max()
print('Piekvermogen te beschouwen in de berekening is '+str(piekMei21)+'kW')
verbMei21=dataA41[(dataA41.index.month==5)]['Vermogen_kW'].sum()*.25/1000
print('Verbruik te beschouwen in de berekening is '+str(verbMei21)+'MWh')
        ''')
        
st.markdown('Piekvermogen te beschouwen in de berekening is '+str(piekMei21)+'kW \n'+'Verbruik te beschouwen in de berekening is '+str(verbMei21)+'MWh')




