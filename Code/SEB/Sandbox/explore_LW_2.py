#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Here we test how easily LW down can be modelled from other params
"""
import numpy as np, pandas as pd, statsmodels.api as sm, GeneralFunctions as GF
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import minimize

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

""" Below are functions to estimate LW using the method of Remco de Kok et al. 
(2019):
 (https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.6249) -- eq.8:
     
     LW=c1+c2RH+c3*boltz*Tk^4
     
The set of functions includes optimizations to find the coefficients 
"""


def sim_lw(rh,tk,lw,toa):
    
    """
    This is the main coordinating function to estimate LW from RH and T. It 
    calls sub routines to optimize coefficients on subsets of data (cloudy/
    clear). These subsets are found with ToA and Rh
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - rh (%)         : relative humidity
        - tk (K)         : air temperature
        - lw (W/m^2)     : measured longwave radiation (to optimize against)
        - toa (W/m^2)    : top-of-atmosphere insolation
     
    Out:
        - lw_out (W/m^2) : modelled lonwave radiation
             
    """    
    
    day_idx=toa>0
    night_idx=toa==0
    clear_idx=np.logical_or(np.logical_and(day_idx,rh<60),\
                        np.logical_and(night_idx,rh<80))
    cloudy_idx=np.logical_or(np.logical_and(day_idx,rh>=60),\
                        np.logical_and(night_idx,rh>=80))
    
    x0=np.array([1.0,1.0,1.0])
    fit_clear=minimize(lw_rdk,x0,args=(rh[clear_idx],tk[clear_idx],lw[clear_idx]))
    x_clear=fit_clear.x
    fit_cloudy=minimize(lw_rdk,x0,args=(rh[cloudy_idx],tk[cloudy_idx],lw[cloudy_idx]))
    x_cloudy=fit_cloudy.x
    
    lw_mod=np.zeros(len(lw))*np.nan
    lw_mod[clear_idx]=_lw_rdk(x_clear,rh[clear_idx],tk[clear_idx])
    lw_mod[cloudy_idx]=_lw_rdk(x_cloudy,rh[cloudy_idx],tk[cloudy_idx])
    
    return lw_mod

def optimize_lw_rdk(params,rh,tk,lw):
    
    c1=params[0]
    c2=params[1]
    c3=params[2]
    
    lw_mod=_lw_rdk([c1,c2,c3],rh,tk)
    
    err=GF.RMSE(lw_mod,lw)
    
    return err
      
def _lw_rdk(params,rh,tk):
    
    c1=params[0]
    c2=params[1]
    c3=params[2]
    
    lw_mod=c1+c2*rh+c3*boltz*tk**4   
    
    return lw_mod

def RMSE(obs,sim):
    n=np.float(np.sum(~np.isnan((obs+sim))))
    rmse=np.nansum((obs-sim)**2)/n
    return rmse



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
t=data["T_HMP"]; t=t+273.15
data['RH'] = data['RH'].astype(np.float)
rh=data["RH"].values[:]
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

sim=sim_lw(rh,tk,lw,toa)

# Inspect 
fig,ax=plt.subplots(1,1)
ax.scatter(lw,sim,s=5,color="k")
x=np.linspace(100,350)
ax.plot(x,x,color='red')

