#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Here we test how easily LW down can be modelled from other params
"""
import numpy as np, pandas as pd, statsmodels.api as sm, GeneralFunctions as GF
import matplotlib.pyplot as plt

""" Below are functions to compute potential (ToA) solar radiation. These
functions are used to figure out day/night for purposes of setting night-time 
sin values to zero. """

# Functions (also now in GF)
def _decl(lat,doy):
    # Note: approzimate only. Assumes not a leap year
    c=2.*np.pi
    dec=np.radians(23.44)*np.cos(c*(doy-172.)/365.25)

    return dec, np.degrees(dec)

def _sin_elev(dec,hour,lat,lon):
    c=c=2.*np.pi
    lat=np.radians(lat)
    out=np.sin(lat)*np.sin(dec) - np.cos(lat)*np.cos(dec) * \
    np.cos(c*hour/24.+np.radians(lon))
    
    # Out is the sine of the elevation angle
    return out, np.degrees(out)

def _sun_dist(doy):
    c=c=2.*np.pi
    m=c*(doy-4.)/365.25
    v=m+0.0333988*np.sin(m)+0.0003486*np.sin(2.*m)+0.0000050*np.sin(3.*m)
    r=149.457*((1.-0.0167**2)/(1+0.0167*np.cos(v)))
    
    return r
    

def sin_toa(doy,hour,lat,lon):
    dec=_decl(lat,doy)[0]
    sin_elev=_sin_elev(dec,hour,lat,lon)[0]
    r=_sun_dist(doy)
    r_mean=149.6
    s=1366.*np.power((r_mean/r),2)
    toa=sin_elev*s; toa[toa<0]=0.
    
    return toa, r

# # # # # # # # # # # # # # #
# Parameters and constants
# # # # # # # # # # # # # # #
fin="/home/lunet/gytm3/Everest2019/AWS/Logging/south_col.csv"
rad_thresh=10. # W/m**2 --> used to help identify periods of snow-covered
# radiation sensors
boltz=5.67*10**-8
inc=2.

# Read in and pre-process
data=pd.read_csv(fin,sep=",",parse_dates=True, index_col=0,keep_default_na=False,\
                 na_values="NaN")
# Convert to floats 
data["SW_IN_AVG"]=data['SW_IN_AVG'].astype(np.float)
sw=data["SW_IN_AVG"].values[:]
data['T_HMP'] = data['T_HMP'].astype(np.float)
t=data["T_HMP"]
data['RH'] = data['RH'].astype(np.float)
data['LW_IN_AVG'] = data['LW_IN_AVG'].astype(np.float)
data['SW_OUT_AVG'] = data['SW_OUT_AVG'].astype(np.float)
# Remaining post-processing
data["SW_IN_AVG"].values[data["SW_IN_AVG"].values[:]<0]=0
sw_o=data["SW_OUT_AVG"].values[:]
toa,r=sin_toa(data.index.dayofyear.values[:],\
              data.index.hour.values[:],27.98,86.93)
sw[toa==0]=0.0
sw_o[toa==0]=0.0
# Find where SW_IN < SW_OUT. This are likely snow-covered periods. 
sw_idx=sw<sw_o

# Filter rad data where snow is likely on the upward-looking sensors
idx=np.logical_and(data["SW_IN_AVG"]>rad_thresh,data["SW_OUT_AVG"]>\
                   data["SW_IN_AVG"])
data["SW_IN_AVG"].values[idx]=np.nan
data["LW_IN_AVG"].values[idx]=np.nan
lw=data["LW_IN_AVG"].values[:]
data["TK"]=data["T_HMP"]+273.15
data["VP"]=GF.satVp(data["TK"])*data["RH"]/100.

# # # # # # # # # # # # # # #
# Start fitting
# # # # # # # # # # # # # # #

# Fit clear-sky LW function in 5k bins:

mint=np.min(t); maxt=np.percentile(t,95)
ts=np.arange(mint,maxt,inc)
out=np.zeros((len(ts),2))
count=0
for ti in ts:
    idx=np.logical_and(t>=ti,t<ti+inc)
    out[count,0]=(ti+ti+inc)/2.+273.15; out[count,1]=np.nanpercentile(lw[idx],5)
    count+=1
fig,ax=plt.subplots(1,1)
ax.scatter(out[:,0],out[:,1])
assert 1==2

# Compute emissivity 
e=data["LW_IN_AVG"]/(boltz*data["TK"]**4)
e.values[e>1]=np.nan
#e.plot()

# Fit regression
sw_thresh=0.8*np.nanmax(data["SW_IN_AVG"].values[:])
X=np.column_stack((data["VP"].values[:],data["RH"].values[:]))
X=sm.add_constant(X)
Y=e.values[:]
idx=np.logical_and(np.logical_and(~np.isnan(X[:,1]),~np.isnan(X[:,2])),\
                   ~np.isnan(Y))
idx=np.logical_and(idx,data["SW_IN_AVG"]>sw_thresh)
est = sm.OLS(e.values[idx], X[idx,:-1]).fit()
pred = est.predict(X[idx,:-1])
fig,ax=plt.subplots(1,1)
ax.scatter(pred,Y[idx])
ax.set_xlim(0.5,1)
ax.set_ylim(0.5,1)
x=np.linspace(0.5,1,100)
ax.plot(x,x)

fig,ax=plt.subplots(1,1)
ax.scatter(data["VP"],e,s=4)

