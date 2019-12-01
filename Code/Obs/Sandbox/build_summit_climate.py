#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script estimates the summit climate -- for the purpose of computing
the SEB up there. 

Here we use a 1st order polynomial to model 
    
    - transmissivity &
    - rh

as a function of height, using data fro Phortse, C2, South Col and Balcony
(where data are available)    

Air temperature is adjusted by reading in the lapse rates in: 
    
/home/lunet/gytm3/Everest2019/AWS/Research/BAMS/Data/AWS/Temp_Lapse.csv"

Winds will be taken from the South Col or Balcony (whichever is windier)
and the LW model coefficients are taken from the South Col

Pressure from the Balcony will be adjusted to the summit using the 
hypsometric equation and lapse rates
    
"""
import datetime, pandas as pd, numpy as np
from numba import jit
import scipy.optimize as optimize
from scipy.optimize import minimize
import GeneralFunctions as GF
import matplotlib.pyplot as plt


# # # # # # # # # # # # 
# Constants
# # # # # # # # # # # # 
g=8.80655 # gravitational acceleration
Rd=287. # gas constant for dry air


# # # # # # # # # # # # 
# Functions
# # # # # # # # # # # # 
@jit
def _decl(lat,doy):
    # Note: approzimate only. Assumes not a leap year
    c=2.*np.pi
    dec=np.radians(23.44)*np.cos(c*(doy-172.)/365.25)

    return dec, np.degrees(dec)

@jit
def _sin_elev(dec,hour,lat,lon):
    c=c=2.*np.pi
    lat=np.radians(lat)
    out=np.sin(lat)*np.sin(dec) - np.cos(lat)*np.cos(dec) * \
    np.cos(c*hour/24.+np.radians(lon))
    
    # Out is the sine of the elevation angle
    return out, np.degrees(out)

@jit
def _sun_dist(doy):
    c=c=2.*np.pi
    m=c*(doy-4.)/365.25
    v=m+0.0333988*np.sin(m)+0.0003486*np.sin(2.*m)+0.0000050*np.sin(3.*m)
    r=149.457*((1.-0.0167**2)/(1+0.0167*np.cos(v)))
    
    return r
    
@jit
def sin_toa(doy,hour,lat,lon):
    dec=_decl(lat,doy)[0]
    sin_elev=_sin_elev(dec,hour,lat,lon)[0]
    r=_sun_dist(doy)
    r_mean=149.6
    s=1366.*np.power((r_mean/r),2)
    toa=sin_elev*s; toa[toa<0]=0.
    
    return toa, r

def adjustT(temp,lapse,dz):
    """
    Simple function to take an observed temperature (temp) and adjust for a 
    change in elevation (dz) using the lapse rate (lapse)
    
    *** NB: Lapse in degC or K per m! *** 
    """
    temp_adj=temp+lapse*dz
    
    return temp_adj

def adjustP(press,temp,lapse,dz):
    """
    Simple function using the hypsometric equation to estimate the pressure 
    after a change in altitude of dz, given starting pressure (press) and 
    starting temp (temp)
    """
    if np.max(temp)<100: _temp =temp+273.15# NB: C-->K
    tav=1/2.*(_temp+adjustT(_temp,lapse,dz))
    press_adj=press*np.exp(-dz*g/(Rd*tav))
    
    return press_adj

def RHO(p,tv):
    
    """
    Computes the air density
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)  : air pressure
        - tv (K)  : virtual temperature (can be approximated as T if air is
                    very dry -- low specific humidity)
        
    Out:
    
        - rho (kg/m^3) : air density
        
    """    
    if np.max(tv)<100: tv +=273.15# NB: C-->K
    if np.max(p)<2000: p*=100 # hPa to Pa
    rho=np.divide(p,np.multiply(Rd,tv))
    
    return rho

def temp_uncertainty(se,n,x0,mu_x,denom):

    """ Computes the 95% uncertainty with the se^2, x0 and sum(xi-mu(x))^2
    as input (i.e. denominator). See eq. 7.22 of Wilks (2011) for details, 
    and x-ref to temp_lapse.py"""
    
    sy=1.96*np.sqrt(se*(1+1/n+np.power(x0-mu_x,2)/denom))
    
    return sy

""" Below are functions to estimate LW using the method of Remco de Kok et al. 
(2019):
 (https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.6249) -- eq.8:
     
     LW=c1+c2RH+c3*boltz*Tk^4
     
The set of functions includes optimizations to find the coefficients 
"""

def sim_lw(rh,tk,toa,x_cloudy,x_clear):
    
    """
    This function is the same as 'fit_lw', except it just models lw -- given 
    the optimized coefficients
     
    Inputs/Outputs (units: explanation): 
    
    In:
        - rh (%)         : relative humidity
        - tk (K)         : air temperature
        - lw (W/m^2)     : measured longwave radiation (to optimize against)
        - toa (W/m^2)    : top-of-atmosphere insolation
     
    Out:
        - lw_out (W/m^2) : modelled lonwave radiation
             
    """    
    if np.min(tk)<100: _tk=tk+273.15 # C-->K
    
    day_idx=toa>0
    night_idx=toa==0
    clear_idx=np.logical_or(np.logical_and(day_idx,rh<60),\
                        np.logical_and(night_idx,rh<80))
    cloudy_idx=np.logical_or(np.logical_and(day_idx,rh>=60),\
                        np.logical_and(night_idx,rh>=80))
        
    lw_mod=np.zeros(len(toa))*np.nan
    lw_mod[clear_idx]=_lw_rdk(x_clear,rh[clear_idx],_tk[clear_idx])
    lw_mod[cloudy_idx]=_lw_rdk(x_cloudy,rh[cloudy_idx],_tk[cloudy_idx])
    
    return lw_mod


def fit_lw(rh,tk,lw,toa):
    
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
    if np.min(tk)<100: _tk=tk+273.15  # C-->K
    
    day_idx=toa>0
    night_idx=toa==0
    clear_idx=np.logical_or(np.logical_and(day_idx,rh<60),\
                        np.logical_and(night_idx,rh<80))
    cloudy_idx=np.logical_or(np.logical_and(day_idx,rh>=60),\
                        np.logical_and(night_idx,rh>=80))
    
    x0=np.array([1.0,1.0,1.0])
    fit_clear=minimize(optimize_lw_rdk,x0,\
                       args=(rh[clear_idx],_tk[clear_idx],lw[clear_idx]))
    x_clear=fit_clear.x
    fit_cloudy=minimize(optimize_lw_rdk,x0,\
                        args=(rh[cloudy_idx],_tk[cloudy_idx],lw[cloudy_idx]))
    x_cloudy=fit_cloudy.x
    
    lw_mod=np.zeros(len(lw))*np.nan
    lw_mod[clear_idx]=_lw_rdk(x_clear,rh[clear_idx],_tk[clear_idx])
    lw_mod[cloudy_idx]=_lw_rdk(x_cloudy,rh[cloudy_idx],_tk[cloudy_idx])
    
    return lw_mod, x_cloudy, x_clear

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



# # # # # # # # # # # # 
# Filenames etc
# # # # # # # # # # # #     
di="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
dout="/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/Public/"
fs=["balcony_filled","south_col_filled","c2","kcc_log"]
lapsef=di+"Temp_Lapse.csv"
names=["balcony","south_col","c2","kcc"]
zs=np.array([8430.,7945.,6464.,3810.])
n=len(names)
rad_thresh=100
albedo=0.8
oname=di+"summit_guess.csv"
boltz=5.67*10**-8

# # # # # # # # # # # # 
# Main Code
# # # # # # # # # # # # 

# Can read lapse in here
lapse=pd.read_csv(lapsef,parse_dates=True,index_col=0)

# Loop one 
data={}
for i in range(len(names)):
    scratch=pd.read_csv(di+fs[i]+".csv",parse_dates=True,index_col=0)
    doy=scratch.index.dayofyear.values[:]
    hour=scratch.index.hour.values[:]
    toa,r=sin_toa(doy,hour,27.98,86.93)
    if names[i] != "balcony":
        # Correct sin
        si=scratch["SW_IN_AVG"].astype(np.float)
        so=scratch["SW_OUT_AVG"].astype(np.float)
        toa,r=sin_toa(doy,hour,27.98,86.93)
        si.loc[si<0]=0
        si.loc[toa==0]=0
        run_albedo=so.rolling(24).sum()/si.rolling(24).sum()
        run_albedo.loc[run_albedo>0.9]=0.90
        si[si<so]=so[si<so]/run_albedo[si<so]
        toa=pd.Series(toa,index=scratch.index)
        snow_idx=idx=np.logical_and(si>rad_thresh,so>si)       
        scratch["SNOW_COVER"]=snow_idx
        scratch["LW_IN_AVG"].loc[snow_idx]=np.nan
        scratch["LW_IN_AVG"]=scratch["LW_IN_AVG"].astype(np.float)
        
        # Compute the running transmissivity
        run_trans=si.rolling(24).sum()/toa.rolling(24).sum()
        scratch["TRANS"]=run_trans.interpolate(method="linear").ffill().bfill()
        
        # Fill missing
#    scratch["T_HMP"]=scratch["T_HMP"].astype(np.float).\
#                    interpolate(method="linear").ffill().bfill()
#    scratch["RH"]=scratch["RH"].astype(np.float).\
#                    interpolate(method="linear").ffill().bfill()

    scratch["RUN_T"]=scratch["T_HMP"].rolling(24).mean().interpolate()
    scratch["RUN_RH"]=scratch["RH"].rolling(24).mean().interpolate()
    scratch["PRESS"]=scratch["PRESS"].astype(np.float).interpolate()
    scratch["TOA"]=toa
    data[names[i]]=scratch

# Latest start date
min_date=max([data[names[i]].index[0] for i in range(n)])+datetime.timedelta(days=1)
max_date=datetime.datetime(year=2019,month=11,day=1)

# Outside of loop estimate the coefficients needed to model LW in at the 
# South Col -- using rh and temp
mod,x_cloudy,x_clear=fit_lw(data["south_col"]["RH"],\
                            data["south_col"]["T_HMP"],\
                            data["south_col"]["LW_IN_AVG"],\
                            data["south_col"]["TOA"])

# Get tref (still stored in scratch)
tref=scratch.index[np.logical_and(scratch.index>=min_date,\
                                      scratch.index<=max_date)]

 # Preallocate for the gradients   
trans_grad=np.zeros(len(tref))*np.nan  
temp_grad=np.zeros(len(tref))*np.nan 
rh_grad=np.zeros(len(tref))*np.nan 

# Preallocte for the summit reconstruction
summit_sw=np.zeros(len(tref))*np.nan 
summit_toa=np.zeros(len(tref))*np.nan 
summit_temp=np.zeros(len(tref))*np.nan 
summit_rh=np.zeros(len(tref))*np.nan 
summit_press=np.zeros(len(tref))*np.nan 
summit_wind=np.zeros(len(tref))*np.nan 
summit_lw=np.zeros(len(tref))*np.nan 
summit_ref_ws=np.zeros(len(tref))*np.nan 
summit_ref_z=np.zeros(len(tref))*np.nan 

# Preallocate to store goodness-of-fits
r_trans=np.zeros(len(tref))*np.nan 
r_temp=np.zeros(len(tref))*np.nan 
r_rh=np.zeros(len(tref))*np.nan 
r_press=np.zeros(len(tref))*np.nan 
err_sw=np.zeros(len(tref))*np.nan 
err_temp=np.zeros(len(tref))*np.nan 
err_rh=np.zeros(len(tref))*np.nan 

# Loop 2: iterate over all time steps and compute gradients to 
# guess the summit climate
count=0  
for t in tref:
    
    # sin -- computed with trans from running 24-hour sums of sin/toa
    toa=data["south_col"]["TOA"].loc[data["south_col"].index==t][0]
    y_trans=np.array([data["kcc"]["TRANS"].loc[data["kcc"].index==t][0],\
                     data["c2"]["TRANS"].loc[data["c2"].index==t][0],\
                     data["south_col"]["TRANS"].loc[data["south_col"].index==t][0]])
    
    if (~np.isnan(y_trans)).all(): 
        x_trans=np.array([zs[3],zs[2],zs[1]])
        
        trans_grad[count]=np.polyfit(x_trans,y_trans,1)[0]
        summit_sw[count]=(trans_grad[count]*(8850.-7945)+\
             data["south_col"]["TRANS"].loc[data["south_col"].index==t][0])*\
             data["south_col"]["TOA"].loc[data["south_col"].index==t][0]
             
        # Also paste in the toa
        summit_toa[count]=toa
        # Goodness of fit
        r_trans[count]=np.corrcoef(y_trans,x_trans)[0,1]
        # Prediction intervals
        sy,scratch1,scratch2,scratch3=\
                            GF.pred_intervals(x_trans,y_trans,8850.)
        err_sw[count]=sy*toa
                                
    # temps -- take t from site with max wind, and add lapse * dz to this
    # First -- complile ts into array    
    y_temp=np.array([data["kcc"]["T_HMP"].loc[data["kcc"].index==t][0],\
                     data["c2"]["T_HMP"].loc[data["c2"].index==t][0],\
                     data["south_col"]["T_HMP"].loc[data["south_col"].index==t][0],\
                     data["balcony"]["T_HMP"].loc[data["balcony"].index==t][0]])\
                     .astype(np.float)
    # Now winds into array
    y_ws=np.array([data["kcc"]["WS_AVG"].loc[data["kcc"].index==t][0],\
                     data["c2"]["WS_AVG"].loc[data["c2"].index==t][0],\
                     data["south_col"]["WS_AVG"].loc[data["south_col"].index==t][0],\
                     data["balcony"]["WS_AVG_1"].loc[data["balcony"].index==t][0]])\
                     .astype(np.float)
    # max wind and corresponding t/z
    base_ws=np.nanmax(y_ws)
    base_t=y_temp[y_ws==base_ws][0]
    base_z=zs[::-1][y_ws==base_ws][0]
    temp_grad[count]=lapse["lapse"].loc[lapse.index==t].values[0]
    # Extrapolate to the summit from the correct height
    summit_temp[count]=base_t+((8850.0-base_z)*\
               temp_grad[count])

    # Save the z and ws at the location for base_t
    summit_ref_ws[count]=base_ws
    summit_ref_z[count]=base_z
    
    # Compute uncertainty
    se=lapse["se"].loc[lapse.index==t].values[0]
    n=lapse["n"].loc[lapse.index==t].values[0]
    mu_x=lapse["mu_elev"].loc[lapse.index==t].values[0]
    denom=lapse["denom"].loc[lapse.index==t].values[0]
    err_temp[count]=temp_uncertainty(se,n,8850.,mu_x,denom)
          
    # Rh -- again computed with hourly regression of rh on z
    y_rh=np.array([data["kcc"]["RH"].loc[data["kcc"].index==t][0],\
                     data["c2"]["RH"].loc[data["c2"].index==t][0],\
                     data["south_col"]["RH"].loc[data["south_col"].index==t][0],\
                     data["balcony"]["RH"].loc[data["balcony"].index==t][0]])\
                    .astype(np.float)
                    
    if (~np.isnan(y_rh)).all(): 
        x_rh=np.array([zs[3],zs[2],zs[1],zs[0]])
        ps=np.polyfit(x_rh,y_rh,1)
        rh_grad[count]=ps[0]
        summit_rh[count]=np.max([0,np.min([100.,np.polyval(ps,8850)])])
        # Goodness of fit
        r_rh[count]=np.corrcoef(y_rh,np.array(x_rh))[0,1]
        # Pred intervals
        sy,scratch1,scratch2,scratch3=\
                            GF.pred_intervals(x_rh,y_rh,8850.)
        err_rh[count]=sy
        
    
    # Summit pressure uses the lapse rate and balcony pressure to enter in 
    # to the hypsometric equation
    # Note, function template is: "adjustP(press,temp,lapse,dz)", with temp
    # in C or K, lapse in degC or K per m and dz in m 
    summit_press[count]=adjustP(\
                data["balcony"]["PRESS"].loc[data["balcony"].index==t][0],\
                data["balcony"]["T_HMP"].loc[data["balcony"].index==t][0],\
                temp_grad[count],(8850.0-8430))
    
    # Compute density (no need to store)
    rho_summit=RHO(summit_press[count],summit_temp[count])
    rho_col=RHO(data["south_col"]["PRESS"].loc[data["south_col"].index==t][0],\
                data["south_col"]["T_HMP"].loc[data["south_col"].index==t][0])
    rho_bal=RHO(data["balcony"]["PRESS"].loc[data["balcony"].index==t][0],\
                data["balcony"]["T_HMP"].loc[data["balcony"].index==t][0])
    
    # And increase the wind to reflect the drop in pressure. 
    # Take the maximum wind at the south col and balcony and apply this to 
    # the summit
    south_col_wind=np.max([data["south_col"]["WS_AVG"].\
                                  loc[data["south_col"].index==t][0],\
                                  data["south_col"]["WS_AVG_2"].\
                                  loc[data["south_col"].index==t][0]])
    
    balcony_wind=np.max([data["balcony"]["WS_AVG_1"].\
                                  loc[data["balcony"].index==t][0],\
                                  data["balcony"]["WS_AVG_2"].\
                                  loc[data["balcony"].index==t][0]])
    # Take the max of the two
    summit_wind[count]=np.max([south_col_wind,balcony_wind])
    
    # Increase the wind to reflect the drop in density -- scalar value depends 
    # whether South Col or Balcony had faster winds
    if south_col_wind > balcony_wind: scalar= rho_col/rho_summit
    else: scalar = rho_bal/rho_summit
    
    # Upadate to reflect drop in density
    summit_wind[count]*=scalar
    
    if summit_wind[count] >0:
        assert np.logical_and(summit_wind[count] > \
                          south_col_wind,summit_wind[count]>balcony_wind)
            
  
    count+=1
  

# We can compute summit lw outside the main loop
summit_lw=sim_lw(summit_rh,summit_temp,summit_toa,x_cloudy,x_clear)

# Get the LW errors at the South Col
err_lw=np.abs((mod-data["south_col"]["LW_IN_AVG"]).values[:])

# Now write out the summit climate so that it matches the other files 
# (and hence can be easily read into the SEB routine)
out={"T_HMP":summit_temp,"T_109":summit_temp,"RH":summit_rh,
     "PRESS":summit_press,"WS_AVG":summit_wind,"SW_IN_AVG":summit_sw,\
     "SW_OUT_AVG":summit_sw*albedo,"LW_IN_AVG":summit_lw,"LAPSE":temp_grad,\
     "REF_Z":summit_ref_z,"REF_WS":summit_ref_ws,\
     "T_err":err_temp,"SW_err":err_sw,"RH_err":err_rh}

#err_sw=np.zeros(len(tref))*np.nan 
#err_temp=np.zeros(len(tref))*np.nan 
#err_rh=np.zeros(len(tref))*np.nan 

outDF=pd.DataFrame(out,index=tref)
outDF.to_csv(oname,sep=",",float_format="%.6f")


##Make plot of errors, annotated with median error
fig,ax=plt.subplots(2,2)
ax.flat[0].hist(err_temp[~np.isnan(err_temp)],bins=30,facecolor="grey",edgecolor="k"); \
    ax.flat[0].set_xlabel("Temperature ($^{\circ}$C)")
ax.flat[1].hist(err_rh[~np.isnan(err_rh)],bins=30,facecolor="grey",edgecolor="k");\
    ax.flat[1].set_xlabel("Rel. Humidity (%)"); ax.flat[1].set_xlim(0,100)
ax.flat[2].hist(err_sw[err_sw>0],bins=30,facecolor="grey",edgecolor="k");\
    ax.flat[2].set_xlabel("Insolation (W m$^{-2}$)")
ax.flat[3].hist(err_lw[~np.isnan(err_lw)],bins=30,facecolor="grey",edgecolor="k"); \
    ax.flat[3].set_xlabel("Longwave rad. (W m$^{-2}$)") 
plt.tight_layout()             
for i in ax.flat: i.grid()
# Get medians 
med_errs=np.zeros(4)
items=["temp","rh","sw","lw"]
scratch=[err_temp,err_rh,err_sw[err_sw>0],err_lw]
count=0
for i in range(len(scratch)):
    med_errs[i]=np.nanmedian(scratch[i])
    ax.flat[i].axvline(med_errs[i],linestyle="--",color="red")
    print items[i], " = ", med_errs[count]
    count+=1
fig.savefig(dout+"Summit_climate_uncertainty.png",dpi=300)
