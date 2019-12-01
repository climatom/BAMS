#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This simple script plots the GEV distributions (and computes return levels)
for the (ranalysis) summit/south col climates
"""
import scipy.stats as stat
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Percentile to plot for 
pc=100

# Read in the coefficients
coefs=np.loadtxt("GEV_coefs.csv",delimiter=",")
locs=coefs[:,0]; scales=coefs[:,1]; shapes=coefs[:,2]

# Read in the data
data=pd.read_csv("ann_max.csv")
year = np.arange(1979,2019)

# Read in the raw time series
raw_data=pd.read_csv("extreme_series.csv"); 
index=pd.date_range(start="1/1/1979 00:00:00", end="31/12/2018 23:00:00",freq="6h")
raw_data.index=index
clim_extreme=np.zeros((12))

for m in range(1,13):
    clim_extreme[m-1]=np.nanpercentile\
    (raw_data["ws_sum"].loc[raw_data.index.month==m],99.)

std=np.zeros((40,12))    
for y in range(1979,2019):
    for m in range(1,13):
        idx=np.logical_and(index.month==m, index.year==y)
        std[y-1979,m-1]=np.percentile(raw_data["ws_sum"].loc[idx],99.)
std=np.std(std,axis=0)

# Create pdf reference for temp (-50 --> -30)
temp_x_summit=np.linspace(-50,-37.5,100)
temp_x_col=np.linspace(-45,-32.5,100)

# Ditto for wind speed (50 --> 80)
ws_x_summit=np.linspace(50,90,100)
ws_x_col=np.linspace(40,75,100)

# Plot
fig,ax=plt.subplots(3,1)
fig.set_size_inches(5,8)
# Temp summit
#pdf_temp_summit=\
#stat.genextreme.pdf(temp_x_summit,shapes[0],loc=-locs[0],scale=scales[0])
#ax.flat[0].plot(temp_x_summit,pdf_temp_summit)
## Temp Col
#pdf_temp_col=\
#stat.genextreme.pdf(temp_x_col,shapes[1],loc=-locs[1],scale=scales[1])
#ax.flat[0].plot(temp_x_col,pdf_temp_col)
## Ws summit
#pdf_ws_summit=\
#stat.genextreme.pdf(ws_x_summit,shapes[2],loc=locs[2],scale=scales[2])
#ax.flat[1].plot(ws_x_summit,pdf_ws_summit)
## Ws col
#pdf_ws_col=\
#stat.genextreme.pdf(ws_x_col,shapes[3],loc=locs[3],scale=scales[3])
#ax.flat[1].plot(ws_x_col,pdf_ws_col)

# Compute percentiles
pc_temp_sum=-np.percentile(data["y_sum"],pc); print pc_temp_sum
pc_temp_col=-np.percentile(data["y_col"],pc); print pc_temp_col
pc_ws_sum=np.percentile(data["ws_sum"],pc); print pc_ws_sum
pc_ws_col=np.percentile(data["ws_col"],pc); print pc_ws_col

# Plot
ax.flat[0].plot(year,-data["y_sum"],color="black",marker='.')
ax.flat[0].plot(year,-data["y_col"],color="grey",marker='.')
ax.flat[1].plot(year,data["ws_sum"],color="black",marker='.',label="Summit (8,848 m)")
ax.flat[1].plot(year,data["ws_col"],color="grey",marker='.',label="South. Col (8,000 m)")
ax.flat[0].set_xticklabels([])
ax.flat[0].axhline(pc_temp_sum,color="k",linestyle="--")
ax.flat[0].axhline(pc_temp_col,color="grey",linestyle="--")
ax.flat[1].axhline(pc_ws_sum,color="k",linestyle="--")
ax.flat[1].axhline(pc_ws_col,color="grey",linestyle="--")
ax.flat[0].set_ylabel("Temperature ($^{\circ}$C)")
ax.flat[1].set_ylabel("Wind speed (m/s)")
ax.flat[1].set_ylim(42.5,80)
ax.flat[1].legend()
months=np.arange(1,13)
ax.flat[2].fill_between(months,clim_extreme-std,clim_extreme+std,color="k",\
       alpha=0.25)
ax.flat[2].plot(months,clim_extreme,color="k",marker=".",markersize=10)
ax.flat[2].set_ylabel("Wind speed (m/s)")
ax.flat[2].set_xlabel("Month")
ax.flat[2].set_xticks([2,4,6,8,10,12])
plt.tight_layout()
fig.savefig("Reanalysis_temp_wind.png",dpi=300)