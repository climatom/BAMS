#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script analyses everest time series (for trends/window analysis, etc)

More details to follow!
"""
import numpy as np, pandas as pd
import datetime
import matplotlib.pyplot as plt

def calc_run(onezero):
    # Computes the length of a run of ones
    out=np.zeros(len(onezero))
    scratch={}
    acc=0
    for i in range(1,len(onezero)):
        year=onezero.index.year[i]
        
        if year not in scratch.keys():
            scratch[year]=[]
            
        if onezero.index.day[i]==1:
            out[i]=np.nan
            acc=0
            continue
        
        elif onezero[i]==0:
            if onezero[i-1]==1:
                scratch[year].append(out[i-1])
            acc=0
            
        else: 
            acc+=1
            if onezero.index[i].day == 31:
                scratch[year].append(acc) # needed if we don't end the run
                # in May!
        
        out[i]=acc
        
    # Now iterate over years
    years=np.unique(onezero.index.year)
    out_year = np.zeros(len(years))
    for i in range(len(years)):
        out_year[i]=np.nanmax(scratch[years[i]])
        
    return out, out_year, scratch

# Name of text file with interpolated reanalysis data
fin="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/summit.txt"

# Name of summiters' file
fin_sum="/home/lunet/gytm3/Everest2019/Research/Window_Analysis/summits.xlsx"
summits=pd.read_excel(fin_sum,parse_dates=True,index_col="msmtdate")

## Which month to extract
month=5

# Read in with pandas
data=pd.read_csv(fin,parse_dates=True,index_col=0,sep="\t")
# Change time to NST
data.index=data.index+datetime.timedelta(hours=5.75)

# Compute daily mean summit climate (for successful summit days)
daily_mean=data.resample("D").mean()
daily_min=data.resample("D").mean()
daily_max=data.resample("D").max()

# Create summit subset of wind_max
idx=daily_mean.index.isin(summits.index)
# Percentiles for max wind and min temp
pc90_wind_max=np.percentile(daily_max["ws"].loc[idx],90)
pc90_wind_mean=np.percentile(daily_mean["ws"].loc[idx],90)
print pc90_wind_max, pc90_wind_mean

# Extract month of interest
data_sel=daily_mean.loc[daily_mean.index.month==month]
data_sel["flag"]=(data_sel["ws"]<=pc90_wind_mean)*1.

# Compute 'runs' of good weather
# out_year is the max window length in a given year. It is the most interesting
out, out_year,scratch=calc_run(data_sel["flag"])
data_sel["runs"]=out

