#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This code computes the lapse rate for temperature from KCC, Camp II, 
South Col, and Balcony. 

Note that temperatures are first smoothed with 24-hour 
running mean.

"""
import datetime, pandas as pd, numpy as np


def se(x,y):
    
    n=np.float(len(x))
    yi=np.polyval(np.polyfit(x,y,1),x)
    _se=1./(n-2)*np.sum(np.power(y-yi,2))
    
    return _se

# Params
din="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
dout="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
fins=["kcc_log.csv","c2.csv","south_col_filled.csv","balcony_filled.csv"]
fout=dout+"Temp_Lapse.csv"
fout_temps=dout+"Mutual_Temps.csv"

elevs=np.array([3810.,6464.,7945.,8430.])

# Find min and max dates
st=datetime.datetime.today()-datetime.timedelta(days=10000)
end=datetime.datetime.today()+datetime.timedelta(days=10) 
for f in fins:
    data=pd.read_csv(din+f,parse_dates=True,index_col=0)
    st=np.max([st,data.index[0]])
    end=np.min([end,data.index[-1]])
n=np.int((end-st).total_seconds()/3600.+1)    

# Now repeat -- actually read in the data, truncate, and put in to array
tarray=np.zeros((n,4))*np.nan
lapse=np.zeros(tarray.shape[0])*np.nan
se_out=np.zeros((tarray.shape[0],4))*np.nan
i=0
for f in fins:
    data=pd.read_csv(din+f,parse_dates=True,index_col=0)
    temp=data["T_HMP"].loc[np.logical_and(data.index>=st,\
             data.index<=end)]
    tarray[:,i]=temp.rolling(24).mean()
    i+=1
    
# Now compute the lapse
for i in range(len(tarray)):
    idx=~np.isnan(tarray[i,:])
    if np.sum(idx)>1:
        lapse[i]=np.polyfit(elevs[idx],tarray[i,idx],1)[0]
        se_out[i,0]=se(elevs[idx],tarray[i,idx])
        se_out[i,1]=np.sum(idx)
        se_out[i,2] = np.sum((elevs[idx]-np.nanmean(elevs))**2.0)
        se_out[i,3] = np.mean(elevs[idx])

# Write the lapse rate out
out=pd.DataFrame(data={"lapse":lapse,"se":se_out[:,0],"n":se_out[:,1],\
                       "denom":se_out[:,2],\
                       "mu_elev":se_out[:,3]},\
                 index=data.index)
out.to_csv(fout,sep=',',float_format="%.6f")

# Also write the 'mutual temps' array out (handy for computing things like 
# the mean 'mountain temp')
out=pd.DataFrame({"kcc":tarray[:,0],\
                  "c2":tarray[:,1],\
                  "scol":tarray[:,2],\
                  "bal":tarray[:,3]},index=data.index)
out.to_csv(fout_temps,sep=',',float_format="%.3f")

print "Lapse rates range between %.1f and %.1f" % (np.nanmin(lapse*1000.),\
                                                   np.nanmax(lapse*1000.))
    
print "Phortse on average %.0f degC warmer than Balcony over common period" %\
(np.nanmean(tarray[:,0])-np.nanmean(tarray[:,-1]))



