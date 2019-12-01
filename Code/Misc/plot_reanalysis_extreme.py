#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This cript plots the reanalysis summit / south col extremes
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'Sans',
        'size'   : 7}

plt.rc('font', **font)

# Import / define
dout="/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/Public/"
fin="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/Reanalysis/extreme_series.csv"
data=pd.read_csv(fin,index_col=0)


date_ref=pd.date_range(start="1/1/1979 00:00",end="31/12/2018 18:00",freq="6H")
data.set_index(date_ref,inplace=True)
wind_max=data[["ws_sum","ws_col"]].groupby(data.index.month).max()
temp_min=data[["t_sum","t_col"]].groupby(data.index.month).min()
temp_std=[np.std((data["t_sum"]\
[data.index.month==ii].groupby(data.index.year[data.index.month==ii])).min())\
for ii in range(1,13)] 

wind_std=[np.std((data["ws_sum"]\
[data.index.month==ii].groupby(data.index.year[data.index.month==ii])).max())\
for ii in range(1,13)] 

# Summarise 
print "All-time min temp = %.2f"%data["t_sum"].min()
print "All-time max wind = %.2f"%data["ws_sum"].max()

# Draw axes
fig,ax=plt.subplots(2,2)
fig.set_size_inches(8,5)
# Temp
data["t_sum"].resample("m").min().plot(ax=ax.flat[0],color='k',linewidth=1)
#data["t_col"].plot(ax=ax.flat[0],color='grey',linewidth=0.2)
# Wind Speed
data["ws_sum"].resample("m").max().plot(ax=ax.flat[2],color='k',linewidth=1)
#data["ws_col"].plot(ax=ax.flat[2],color='grey',linewidth=0.2)

# Temp
temp_min["t_sum"].plot(ax=ax.flat[1],color='k',linewidth=1)
ax.flat[1].fill_between(range(1,13),temp_min["t_sum"]-temp_std,\
       temp_min["t_sum"]+temp_std,color='k',alpha=0.3)
#temp_min["t_col"].plot(ax=ax.flat[1],color='grey',linewidth=0.2)
# Wind Speed
wind_max["ws_sum"].plot(ax=ax.flat[3],color='k',linewidth=1)
ax.flat[3].fill_between(range(1,13),wind_max["ws_sum"]-temp_std,\
       wind_max["ws_sum"]+wind_std,color='k',alpha=0.3)
#wind_max["ws_col"].plot(ax=ax.flat[3],color='grey',linewidth=0.2)

# Tidy
ax.flat[0].axhline(temp_min["t_sum"].min(),color='red')
ax.flat[2].axhline(wind_max["ws_sum"].max(),color='red')
ax.flat[0].set_ylabel("Min. Air Temperature ($^{\circ}$C)")
ax.flat[2].set_ylabel("Max. Wind Speed (m s$^{-1}$)")
ax.flat[2].set_xlabel("Year")
ax.flat[3].set_xlabel("Month")
for ii in ax.flat: ii.grid()
fig.savefig(dout+"Reanalysis_Extremes.png",dpi=300)