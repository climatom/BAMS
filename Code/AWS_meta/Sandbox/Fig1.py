#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple script to create Fig. 1 in BAMS manuscript (histogram of weather staion
 #no as f(elevation); and of #glacier area as f(elevation))
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# Query elevation (tuning parameter)
query=6000.

# Read in AWS data
aws=pd.read_csv("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/"+\
                "Processed_AWS_All_30112019.csv"); 
n = len(aws)
uel=np.unique(aws["elev"].values[:])
ne=[]
for e in uel:
    ne.append(np.sum(aws["elev"].values[:]>=e)/np.float(n)*100.)
lon_aws=aws["lon"]
lat_aws=aws["lat"]
yr_stp_aws=aws["yrstp"]

# Read in glacier data and allocate
data=np.array(pd.read_csv(\
"/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/asia_glaciers.csv",\
sep=","))
area=data[:,2]
lon_glac=data[:,2]
lat_glac=data[:,3]
hyps=data[:,3:]

# Get lat/lon limits for asia
lonmin=np.floor(np.min(lon_glac)); lonmax=np.ceil(np.max(lon_glac))
latmin=np.floor(np.min(lat_glac)); latmax=np.ceil(np.max(lat_glac))
# Change missing flag to 9999 so that we can include those stations missing 
# this info.
aws.loc[aws["yrstp"]==999]=9999.; 

# Drop all stations with a start in 2019 (includes evernet, and 
# Chinese 2019 install on north side).
aws=aws.loc[~(aws["yrst"]==2019).values[:]]

# Also drop those that are not active. Include Ding et al. (2018) (NaN)
aws=aws.loc[np.logical_or(aws["yrstp"]==2019,np.isnan(aws["yrstp"]))]

# Asian only 
aws_asia=aws.loc\
               [np.logical_and(np.logical_and(lon_aws>=lonmin,lon_aws<=lonmax),\
               np.logical_and(lat_aws>=latmin,lat_aws<=latmax))]

hyps=pd.read_csv\
("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/hyps_all.csv")
uel_a=np.unique(aws_asia["elev"].values[:])
ne_a=[]
for e in uel_a:
    ne_a.append(np.sum(aws_asia["elev"].values[:]>=e))

# Names of regions
names=hyps.columns

# Create figure
fig,ax=plt.subplots(1,1)
fig.set_size_inches(8,9)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,17 )])
elev=aws_asia["elev"]
asia=np.zeros(len(hyps))

# Twin the x-axis twice to make independent y-axes.
axes = [ax, ax.twinx(), ax.twinx()]

# Make some space on the right side for the extra y-axis.
fig.subplots_adjust(right=0.8)

# Move the last y-axis spine over to the right by 20% of the width of the axes
axes[-1].spines['right'].set_position(('axes', 1.2))

# To make the border of the right-most axis visible, we need to turn the frame
# on. This hides the other plots, however, so we need to turn its fill off.
axes[-1].set_frame_on(True)


count=0
for n in names:
    if "mid" in n or "Greenland" in n or "Antarctic" in n: continue
    x=hyps["mid"].values[:]
    y=hyps[n].values[:]
    if "Asia" not in n:
        axes[0].plot(x,y,alpha=0.5,label=n,linestyle="--")
    else: asia+=y
    count+=1
    
xa=np.append(uel_a,[7500]); ne_a.append(0)
uel=np.array(uel); ne=np.array(ne)
axes[2].plot(xa,ne_a,color="red",label="HMA AWSs",linewidth=3)
axes[1].plot(uel[uel<7500],ne[uel<7500],color="black",label="All AWSs",linewidth=3)
axes[1].set_ylim(0,100)
ax.plot(x,asia,label="HMA",linestyle="--",linewidth=3)   
ax.set_xlabel("Elevation (metres above sea level)")

# plot NG (evernet)
ng=pd.read_csv("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/ng.csv")
y_maxs=np.interp(ng["z"].values[:],x,asia)
dummy=np.ones(len(y_maxs))*250.
y_maxs=np.max(np.column_stack((y_maxs,dummy)),axis=1)
y_mins=np.zeros(len(ng))
#ax.vlines(ng["z"].values[:],ymin=y_mins,ymax=y_maxs,color="grey")
places=["Phortse","Base Camp","Camp II","South Col", "Balcony"]
offset=0
for x,y,z in zip(ng["z"],y_maxs,places):
    if "Balcony" in z: offset=200
    ax.annotate(z, xy=(x, 0), xytext=(x-400, y/5+300.+offset),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            )

# Interesting nuggets
above=np.sum(xa[xa>query])
above_pc=above/np.sum(asia)*100
above_n=np.sum(aws_asia["elev"].values[:]>query)
print query, above, above_pc, above_n

axes[0].set_ylim(0,5000)
axes[1].set_xlim(0,8850)
axes[2].set_ylim(0,250)
axes[0].legend(ncol=2)
axes[1].legend(loc=10,frameon=False)
axes[2].legend(loc=5,frameon=False)
axes[0].set_ylabel("Glacier Area (km$^{2}$)")
axes[1].set_ylabel("% of all AWSs")
axes[1].set_xlabel("Elevation (m)")
axes[2].set_ylabel("Number of HMA AWSs")

fig.savefig\
("/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/Public/Fig1.png",dpi=300)