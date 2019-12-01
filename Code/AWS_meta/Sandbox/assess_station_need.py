#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script computes the "AWS need" on a regular lat/lon grid. See notes 
for details
"""

import  numpy as np, GeneralFunctions as GF, os, pandas as pd


# Read in glacier data and allocate
data=np.loadtxt\
("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/asia_glaciers.csv",\
 skiprows=1,delimiter=",")
area=data[:,0]
lon_glac=data[:,1]
lat_glac=data[:,2]
zmin=data[:,3]
zmed=data[:,4]
zmax=data[:,5]
slope=data[:,6]
aspect=data[:,7]
hyps=data[:,8:]

# Get lat/lon limits
lonmin=np.floor(np.min(lon_glac)); lonmax=np.ceil(np.max(lon_glac))
latmin=np.floor(np.min(lat_glac)); latmax=np.ceil(np.max(lat_glac))

# Read in AWS data and allocate 
data=pd.read_csv("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/ISD_out.csv")
lon_aws=data["lon"].values[:]
lat_aws=data["lat"].values[:]
subset=data.loc[np.logical_and(np.logical_and(lon_aws>=lonmin,lon_aws<=lonmax),\
               np.logical_and(lat_aws>=latmin,lat_aws<=latmax))]
subset.to_csv("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS_meta/Asian_AWSs.csv",\
              index=False)



