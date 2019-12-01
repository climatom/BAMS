#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script cleans the South Col and Balcony data to fix the insidious problem
of repeated/old measurements leaking into the summary statistics 

It also cleans the SW and LW radiation measurements to account for periods of 
snow on the upward-facing sensor

"""
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path.replace("Obs","SEB"))
import seb_utils # note: my module!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Functions
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
dateparse_day = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

def trans_freq(ip,freq="D"):
    
    """
    Checks the number of minutes the satellite modem was turned on per freq
    """
    
    nmins=ip.resample(freq).sum()*-1
    
    return nmins

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#
# @jit
def salv_trh(target,batt_thresh,daily_volt,temprh10):
#    
    """
    Checks for repeated rows of ch200 data. Outputs 'out' -- which contains the
    'fraction of repetition'. Also outputs 'out_flag'. This is set to True
    if the value should be discounted.
    
    Also checks 10-min samples of temp/rh and marks as invlid if all measurements
    are the same; otherwise avearges all on days when battV is suspicious
    """
    # Init vars 
    n=len(target)
    corr=target.values[:,:] # temp and rh
    ten_min_rows=np.arange(len(temprh10))
    fail=0
    
    # Loop 
    for ii in range(n):
       
        # Find 10 min samples falling in the previous hour
        target_hour=target.index.hour[ii]-1
        target_doy=target.index.dayofyear[ii]
        
        # Adjust target hour/doy if spanning midnight
        if target_hour<0:
            target_hour=23
            target_doy-=1
                    
        # Get daily voltage
        idxbatt = daily_volt.index.dayofyear==target.index.dayofyear[ii]
        dv = daily_volt.loc[idxbatt][0]
            
        # Do we have a low voltage?
        if dv <batt_thresh:
            
            # Check 10-min data            
            idx10=np.logical_and(temprh10.index.hour==target_hour,\
                             temprh10.index.dayofyear==target_doy)
            if idx10.any():
                
                # Get correct rows
                row_st=min(ten_min_rows[idx10])-1
                row_stp=max(ten_min_rows[idx10])+1
                scratch=temprh10.iloc[row_st:row_stp,:]
                # Get hourly change
                delta=scratch.values[1:,:]-scratch.values[:-1,:]
                # ID those that have changed since last hour
                idx_short=np.zeros(len(scratch)).astype(bool)
                idx_short[1:]=np.logical_or(delta[:,0]!=0,delta[:,1]!=0) # any != 0 rows
    
                # Check for 0 in RH (dignositic of sensor bug - see 
                # Sept 20, 00:00 at Col)
                ntest=np.sum(scratch.values[:,1]<1.)
                if ntest>0: 
                    corr[ii,0]=np.nan; corr[ii,1]=np.nan
                    continue
                
                # If any have changed, average them. 
                if idx_short.any():
                    mu=np.mean(scratch.loc[idx_short]).values[:]
                    corr[ii,0]=mu[0]; corr[ii,1]=mu[1]
                    continue
            else: print "Missing data for: ", target.index[ii]    
            # Get here if we can't fill gap
            fail+=1
            corr[ii,0]=np.nan; corr[ii,1]=np.nan
            
    print "Found %.0f rows that could not be filled" % fail
    return corr, fail

def salvage_press(hourly,hour_name,daily,day_name,batt_thresh):
    
    """
    Takes the hourly pressure sample and checks to see if it has changed 
    since the last hour. If it hasn't, and if daily min battV on that day was
    below batt_thresh, then we should mark that obs as dubious. We will also 
    replace it with the daily average air pressure
    """
    
    idx=np.zeros(len(hourly)).astype(bool)
    fill=np.zeros(len(idx))*np.nan
    # True where difference since last (hour) is zero
    idx[1:]=(hourly[hour_name].values[1:]-hourly[hour_name].values[:-1])==0
    idx[0]=True
    
    # Iterate over all the "trues" and check min battV on those dates
    check=0
    for i in range(len(idx)):
        
        didx=daily.index.dayofyear==hourly.index.dayofyear[i]
        fill[i]=daily[day_name][didx][0]
        
        if (daily[day_name].loc[didx])[0]<batt_thresh:
            
            # Check if already a NaN
            if ~np.isnan(hourly[hour_name][i]): check+=1

        else: 
            idx[i] = False # Change index to false if daily batt thresh
                             # stayed above thresh
            
    # Only the 'true' elements are suspect. 
    out_nan=hourly*1. 
    out_salv=hourly*1.
    out_nan.set_value(idx,hour_name,np.nan) 
    out_salv.set_value(idx,hour_name,fill[idx])
    
    print "Found %.0f new NaNs" % check
    return out_nan, out_salv

def fix_rad(sw,sw_o,lw,rh,tc,toa):
    
    """
    Function csets nighttime sw to 0, then sets it to out / albedo if sw < sw_o. 
    Periods when sw < sw_o are also used as indicators that the lw_in will
    be compromised, so we use the best empirical guess to replace the missing
    value. 

    """
    # Init
    sw_c=sw*1.
    lw_c=lw*1
    
    # Night to zero
    sw.loc[toa==0]=0.0
    sw_c.loc[toa==0]=0.0
    sw_o.loc[toa==0]=0.0
     
    run_albedo=sw_o.rolling(24,center=True).sum()/sw.rolling(24,center=True).sum()
    run_albedo.loc[run_albedo>0.90]=0.90
      
    assert len(run_albedo)==len(sw)==len(sw_o)
    
    # Out > in replaced by albedo estimate 
    sw_idx=sw<sw_o
    print "Adjusting %.0f rad values" % np.sum(sw_idx)
    sw_c.values[sw_idx]=sw_o.values[sw_idx]/run_albedo.values[sw_idx]
            
    # Simulate LW --for those same periods when sw_o > sw
    lw_ref=seb_utils.sim_lw(rh*100.,tc+273.15,lw,toa)
    lw_c.values[sw_idx]=lw_ref[sw_idx] # -> slot in the best guess   
    
    return sw_c, lw_c, sw_o # Corrected 
       
# Battery thresholds at the AWSs
bcol=11.8
bbal=11.7
bcol_pres=11.6 # Lower threshold for 10-min (pressure) measurements
bbal_pres=11.6 # ditto

# File names for the power tables
din="/home/lunet/gytm3/Everest2019/AWS/Logging/"
dout="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/" 
fpower_col=din+"col_power_out.csv"
fpower_bal=din+"bal_power_out.csv"

# File names for the hourly tables
fcol=din+"south_col.csv"
fbal=din+"balcony.csv"
fc2=din+"c2.csv"

# File name for the daily tables
fdaycol=din+"south_col_day.csv"
fdaybal=din+"balcony_day.csv"

# File name for the 10-min tables
f10col=din+"south_col_10.csv"
f10bal=din+"balcony_10.csv"

# Outnames for QA'd data
fout_col_flag=dout+"south_col_flagged.csv"
fout_col_filled=dout+"south_col_filled.csv"
fout_bal_flag=dout+"balcony_flagged.csv"
fout_bal_filled=dout+"balcony_filled.csv"
fout_c2_filled=dout+"c2_filled.csv"

## Read files in
# Power
power_col=pd.read_csv(fpower_col,parse_dates=['date'],index_col=0)
power_bal=pd.read_csv(fpower_bal,parse_dates=['date'],index_col=0)
# Hourly met
col=pd.read_csv(fcol,parse_dates=True,index_col=0)
for i in col.columns: col[i]=col[i].astype(np.float)
bal=pd.read_csv(fbal,parse_dates=True,index_col=0,date_parser=dateparse)
for i in bal.columns: bal[i]=bal[i].astype(np.float)
# Daily met
col_day=pd.read_csv(fdaycol,parse_dates=True,index_col=0,date_parser=dateparse_day)
bal_day=pd.read_csv(fdaybal,parse_dates=True,index_col=0,date_parser=dateparse_day)
# 10-min met
col10=pd.read_csv(f10col,parse_dates=True,index_col=0,date_parser=dateparse)
bal10=pd.read_csv(f10bal,parse_dates=True,index_col=0,date_parser=dateparse)

# Truncate hourly data to limits of daily data
col=col.loc[np.logical_and(col.index>=np.min(col_day.index),\
                           col.index<=np.max(col_day.index))]
bal=bal.loc[np.logical_and(bal.index>=np.min(bal_day.index),\
                           bal.index<=np.max(bal_day.index))]

col_salv=col*1.  
col_flagged=col*1.        
corr,nfail=salv_trh(col[["T_HMP","RH"]],bcol,col_day["BattV_Min"],col10[["AirTC","RH"]])
col_salv[["T_HMP","RH"]]=corr # export this as filled
col_flagged[["T_HMP","RH"]]=corr

# Now update to deal with pressure air pressure (11.6v - col)
col_nan_press, col_salv_press = \
salvage_press(col,"PRESS",col_day,"BP_mbar_Avg",bcol_pres)

# Repeat for the balcony
bal_nan_press, bal_salv_press = \
salvage_press(bal,"PRESS",bal_day,"BP_mbar_Avg",bbal_pres)

# Combine and write out (note: nothing for bal to combine with)
col_salv["PRESS"]=col_salv_press["PRESS"]
col_flagged["PRESS"]=col_nan_press["PRESS"]

# Update col filled to include 109 T measurements
idx=np.isnan(col_salv["T_HMP"])
col_salv["T_HMP"].loc[idx]=col_salv["T_109"].loc[idx]

# Read in c2 and just process pressure (ffill/bfill/interpolate
c2=pd.read_csv(fc2,parse_dates=True,index_col=0)
for i in c2.columns:
    c2[i]=c2[i].astype(np.float)
c2_salv=c2*1.
c2_salv["PRESS"]=c2["PRESS"].interpolate(method='linear').ffill().bfill()

#=============================================================================#
# QA SW and LW at the South Col and C2
#=============================================================================#
# Fix Col radiation first 
toa,r=seb_utils.sin_toa(col.index.dayofyear.values[:],\
                        col.index.hour.values[:],27.98,86.93)
sw_fixed,lw_fixed,sw_o_fixed=fix_rad(col["SW_IN_AVG"],col["SW_OUT_AVG"],\
                          col["LW_IN_AVG"],col["RH"],col["T_HMP"],toa)

col_salv["SW_IN_AVG"]=sw_fixed
col_salv["SW_OUT_AVG"]=sw_o_fixed
col_salv["LW_IN_AVG"]=lw_fixed

# Now C2
toa,r=seb_utils.sin_toa(c2.index.dayofyear.values[:],\
                        c2.index.hour.values[:],27.98,86.93)
sw_fixed,lw_fixed,sw_o_fixed=fix_rad(c2["SW_IN_AVG"],c2["SW_OUT_AVG"],\
                          c2["LW_IN_AVG"],c2["RH"],c2["T_HMP"],toa)

c2_salv["SW_IN_AVG"]=sw_fixed
c2_salv["SW_OUT_AVG"]=sw_o_fixed
c2_salv["LW_IN_AVG"]=lw_fixed

## Write
col_salv.to_csv(fout_col_filled,sep=",",float_format="%.3f")
col_flagged.to_csv(fout_col_flag,sep=",",float_format="%.3f")
bal_nan_press.to_csv(fout_bal_flag,sep=",",float_format="%.3f")
bal_salv_press.to_csv(fout_bal_filled,sep=",",float_format="%.3f")
c2_salv.to_csv(fout_c2_filled,sep=",",float_format="%.3f")