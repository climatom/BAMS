#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to quickly plot precipitation at Phortse -- cumulative and events

"""
import matplotlib.pyplot as plt
import datetime
import pandas as pd

fin="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/kcc_log.csv"
dout="/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/Sandbox/"
data=pd.read_csv(fin,parse_dates=True,index_col=0)

fig,ax=plt.subplots(1,1)
precip=data["PRECIP"].copy()
precip[precip<0]=0
ax.bar(precip.index,precip)
ax2=ax.twinx()
cumprecip=precip.cumsum()
ax2.plot(cumprecip.index,cumprecip,linewidth=2,color='k')
ax.set_ylabel("Precip Intensity (mm h$^{-1}$)")
ax2.set_ylabel("Precip. Total (mm)")
fig.autofmt_xdate()
ax.axvline(datetime.datetime(year=2019,month=7,day=1),color='red')
ax.axvline(datetime.datetime(year=2019,month=10,day=1),color='red')
fig.savefig(dout+"Phortse_Precip.png",dpi=300)

