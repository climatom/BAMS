#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script converts accumulated radiation fluxes to mean values. It then 
merges these radiation fluxes with another met file (only makes sense to the
author!)

c. TM 17/07/2019 
"""

import numpy as np, pandas as pd
import datetime

# Set outname
oname="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/SouthCol_comb.txt"

# Read in
radfile="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/SouthCol_tisr_raw.txt"
turbfile="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/SouthCol.txt"
heads=["name","date","time","flux"]
rad=pd.read_csv(radfile,delim_whitespace=True, skiprows=1,names=heads)
turb=pd.read_csv(turbfile, sep="\t",parse_dates=True,index_col=0)

# Make proper datetime column 
rad["date"]=pd.to_datetime(rad["date"]+\
   ["T" for ii in range(len(rad))]+rad["time"])
rad.set_index(rad["date"],inplace=True)
hour=rad.index.hour
# Idea is now to difference accumulated radiation fields (all but hour
# == 3 or hour == 15)
dt=3*60**2
st_idx=np.logical_or(hour==3,hour==15)
rest_idx=np.logical_and(hour!=3,hour!=15)
d1=rad["flux"].values[:][1:]
d2=rad["flux"].values[:][:-1]
rad["av_flux"]=np.zeros(len(rad))
rad["av_flux"].values[1:]=(d1-d2)/dt
rad["av_flux"].loc[st_idx]=rad["flux"][st_idx]/dt
# All good, but times need to be fixed!
rad_reindex=rad.reindex(turb.index).interpolate("linear")  
turb["toa_rad"]=rad_reindex["av_flux"]
turb.to_csv(oname)


# Could compute theoretical TOA here


# Could compute thermal rad here...
