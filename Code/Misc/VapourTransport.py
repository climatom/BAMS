#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple script to estimate moisture flux through the South Col. 

The principle here is to take the mean spechum and mean wind between South 
Col and Balcony and estimate the vapour flux in this corridoor.  
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

g=9.81 # Gravitational acceleration

def satVpTeten(tempC):
    
    """
    Teten's formula to compute the saturation vapour pressure
    """
    if np.min(tempC) >150:
        tempC-=273.15
    vp=611.2*np.exp((17.67*tempC)/(tempC+243.5))
    
    return vp


def specHum(temp,rh,press):
     
    """ 
    Simply computes the specific humidity (kg/kg) for given temp, relative 
    humidity and air pressure
     
    Input: 
     
        - temp (C or K)
        - rh (fraction): relative humidity
        - press: air pressure (hPa or Pa)
 
    """
    rh=np.atleast_1d(rh)
    temp=np.atleast_1d(temp)
    press=np.atleast_1d(press)
    # Check for rh as frac and convert if in %
    if np.nanmax(rh)>1.5:
        _rh=rh/100.
    else: _rh = rh*1.    
    # Check for Pa and convert if in hPa (note - no checking for other formats)   
    if np.nanmax(press)<2000:
        _press=press*100.
    else: _press=press*1. 
    # Note that satVp() checks if temp is in K...
    svp=satVpTeten(temp)
    Q=_rh*svp*0.622/_press
    
    return Q


# Read in
din="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
fs=["south_col_filled.csv","balcony_filled.csv"]
col=pd.read_csv(din+fs[0],parse_dates=True,index_col=0)
bal=pd.read_csv(din+fs[1],parse_dates=True,index_col=0)
idx_col=col.index.isin(bal.index)
idx_bal=bal.index.isin(col.index)

# Compute specific humidity
q1=specHum(col["T_HMP"],col["RH"],col["PRESS"])
q2=specHum(bal["T_HMP"],bal["RH"],bal["PRESS"])

# Mean spechum
mu_q=1/2.*(q1[idx_col]+q2[idx_bal])

# Winds
u1=(col["WS_AVG"].values[:]+col["WS_AVG_2"].values[:])/2.
u2=(bal["WS_AVG_1"].values[:]+bal["WS_AVG_2"].values[:])/2.

# Mean wind
mu_u=1/2.*(u1[idx_col]+u2[idx_bal])

# Delta p (Pa)
dp=(col.loc[idx_col]["PRESS"]-bal.loc[idx_bal]["PRESS"])*100.

# Compute IVT
ivt=1/g*dp*mu_q*mu_u
ivt_total=np.sum(ivt)*60**2
ivt_total_m3=ivt_total/1000.

# Compute precipitable water (mm)
pw=1/(1000.*g)*mu_q*dp*1000.

fig,ax=plt.subplots(1,1)
ax.plot(dp.index,ivt)