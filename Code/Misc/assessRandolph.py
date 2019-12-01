#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script analyses the Randolph Glacier Inventory for Asia. 

More detail to be added
"""
import  numpy as np, GeneralFunctions as GF,  pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from netCDF4 import Dataset

fin="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/Misc/asia_glaciers.csv"
crit_z=5000
data=np.array(pd.read_csv(fin,sep=","))
area=data[:,2]
lon_glac=data[:,2]
lat_glac=data[:,3]
hyps=data[:,3:]
# Convert to area above z
area_z=np.array([area*hyps[:,ii]/1000. for ii in range(hyps.shape[1])]).T
area_z_sum=np.nansum(area_z,axis=0)
cum_z=np.cumsum(area_z_sum)
zs=np.arange(25,8850,50)
tot=np.nansum(area)
a=np.sum(area_z_sum[zs>=crit_z])
a_f=a/tot*100.
# Plot
fig,ax=plt.subplots(1,1)
ax.plot(zs,cum_z,color="black")
ax.axvline(crit_z)
ax.set_xlim(0,8850)
ax.set_ylim(0,100000)

print("-----------------------------------------")
print("    Area above %.0fm = %.0f (%.0f%%)"%(crit_z,a,a_f))
print("-----------------------------------------")
