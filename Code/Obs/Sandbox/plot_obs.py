#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Here we plot the initial observations from KCC, Camp II, South Col and 
Balcony. We also compute monthly means/meds/maxs and stdevs (June, July, Aug, Sept)
"""

import datetime, matplotlib.pyplot as plt
import numpy as np, pandas as pd
import matplotlib.dates as mdates

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

def count_valid(arrin):
    return np.sum(~np.isnan(arrin))

#    
# Files
di="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
dout="/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/"
dout_data="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/Misc/"
names=["kcc_log.csv","c2.csv","south_col_filled.csv","balcony_filled.csv"]
clim_names=["kcc","c2.","south_col","balcony.csv"]
files=[di+n for n in names]
lapse=pd.read_csv(di+"Temp_Lapse.csv",parse_dates=True,index_col=0)
rean=pd.read_csv("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/Reanalysis/"+\
                 "extreme_series.csv")
date_ref=pd.date_range(start="01/01/1979 00:00", end="31/12/2018 19:00",freq="6H")
rean.index=date_ref
max_months=rean["ws_sum"].groupby(rean.index.month).max()

# Params
nvalid=18 # Must have this many hours in a day to compute stats. 

# Init plot
fig,ax=plt.subplots(4,1)
fig.set_size_inches(5,11)

# Colours for display
cols=["purple","orange","red","black"]
labels=["Phortse (3810 m)","Camp II (6464 m)","South Col (7945 m)",\
        "Balcony (8430 m)"]

# Monsoon period
end_pre = datetime.datetime(year=2019,month=7,day=1)
end_mons = datetime.datetime(year=2019,month=10,day=1)

# Lims for plotting
min_date=datetime.datetime(year=2019,month=4,day=17)
max_date=datetime.datetime(year=2019,month=11,day=1)

count=0
for f in files:
    
    # Read in data
    data=pd.read_csv(f,parse_dates=True,index_col=0,na_values="NAN")
    data=data.loc[np.logical_and(data.index>=min_date,data.index<max_date)]
    
    # Daily means 
    t=data["T_HMP"];
    t_day=t.resample('D').mean()
    tmin=t.resample('D').min()
    tmax=t.resample('D').max()
    rh=data["RH"]
    rh_day=rh.resample('D').mean()
    press=data["PRESS"]
    
    # Set daily means to nan if computed on <18 hours. 
    ntest=t.resample('D').apply(count_valid)
    t_day.loc[ntest<nvalid]=np.nan
    tmin.loc[ntest<nvalid]=np.nan
    tmax.loc[ntest<nvalid]=np.nan
           
    # Precip (at Phortse)
    if count == 0:
        # Fix the odd precip error (resetting to zero)
        data["PRECIP"].loc[data["PRECIP"]<0]=np.nan        
        precip=np.cumsum(data["PRECIP"])
    
    try:
        ws=1/2.*(data["WS_AVG_1"]+data["WS_AVG_2"])
        ws_max=1/2.*(data["WS_MAX_1"]+data["WS_MAX_2"])

    except:
        try:
            ws=1/2.*(data["WS_AVG"]+data["WS_AVG_2"])
            ws_max=1/2.*(data["WS_MAX"]+data["WS_MAX_2"])
        except:
            ws=data["WS_AVG"]
            ws_max=data["WS_MAX"]
            
            
    # Compute wind speeds - no NaNs for incomplete day as diurnal cycle less
    # prominent 
    ws_day=ws.resample('D').mean()
    ws_max_day=ws_max.resample('D').max()     
    dg_dw,intercept=np.polyfit(ws_day,ws_max_day,1)
    
    r=np.corrcoef(ws_day,ws_max_day)[0,1]
    print "%s gust factor: %.2f; r: %.2f" \
    % (labels[count],dg_dw,r)
    
    if "balcony" not in f: 
        sw=data["SW_IN_AVG"]; 
        
        # Fix sw
        toa_hour,r=sin_toa(sw.index.dayofyear.values[:],\
                        sw.index.hour.values[:],27.98,86.93)
        sw_o=data["SW_OUT_AVG"]
        sw.loc[toa_hour==0]=0.0
        sw_o.loc[toa_hour==0]=0.0
            
        # Find where SW_IN < SW_OUT. Replace with 24-hour-centred albedo-
        # -based estimate. The albedo-based estimate assumes a maximum 
        # possible albedo of 0.91
        run_albedo=sw_o.rolling(24).sum()/sw.rolling(24).sum()
        run_albedo.loc[run_albedo>0.90]=0.90
        sw_idx=sw<sw_o
        sw.loc[sw_idx]=sw_o.loc[sw_idx]/run_albedo.loc[sw_idx]
        # Done!
            
        # Process sw to daily -- it's been cleaned
        sw_day_max=sw.resample('D').max()
        sw_day_mean=sw.resample('D').mean()
        
        # NaN for stats computed on incomplete days
        sw_day_max.loc[ntest<nvalid]=np.nan
        sw_day_mean.loc[ntest<nvalid]=np.nan

        # Plot 
        ax.flat[3].plot(sw_day_max.index,sw_day_max.values[:],\
               color=cols[count])
        
        # Compute monthly stats
        toa_hour=pd.DataFrame(data={"toa":toa_hour},index=sw.index)
        num=sw.rolling(24).sum().values[:]
        denom=np.squeeze(toa_hour.rolling(24).sum().values[:])
        cloud=np.ones(len(denom))-num/denom
        scratch=pd.DataFrame(data=\
        {"temp":t,"ws":ws,"sw":sw,"toa":np.squeeze(toa_hour),"rh":data["RH"].values[:],\
         "cloud":cloud,"press":press,"ws_max":ws_max},index=data.index)
     
    else:
        # Compute monthly stats
        toa_hour,r=sin_toa(sw.index.dayofyear.values[:],sw.index.hour.values[:],\
                   27.99, 86.9250)
        scratch=pd.DataFrame(data=\
        {"temp":t,"ws":ws,"rh":data["RH"].values[:],"press":press,"ws_max":ws_max}\
        ,index=data.index)      
    
    # Write out summary stats
    scratch=scratch.loc[scratch.index.month>5]
    monmean=scratch.resample("M").mean()
    monmax=scratch.resample("M").max()
    monmed=scratch.resample("M").median()
    monstd=scratch.resample("M").std()
    monmean.to_csv(dout_data+clim_names[count]+"_mean.csv",sep=",",\
                   float_format="%.3f")
    monstd.to_csv(dout_data+clim_names[count]+"_std.csv",sep=",",\
                       float_format="%.3f")
    monmed.to_csv(dout_data+clim_names[count]+"_med.csv",sep=",",\
                       float_format="%.3f")
    monmax.to_csv(dout_data+clim_names[count]+"_max.csv",sep=",",\
                       float_format="%.3f")
     
    
    # Now plot 
    ax.flat[0].fill_between(t_day.index,tmin.values[:],\
           tmax.values[:],color=cols[count],alpha=0.25)
    ax.flat[0].plot(t_day.index,t_day[:],color=cols[count],\
           label=labels[count])   
    ax.flat[1].scatter(ws_max_day.index,ws_max_day.values[:],color=\
           cols[count],s=2)
    ax.flat[1].plot(ws_day.index,ws_day.values[:],color=\
           cols[count])    
    # RH
    ax.flat[2].plot(rh_day.index,rh_day.rolling(5).mean().values[:],color=\
           cols[count])   
    
    if count == 0:
        doy=sw_day_max.index.dayofyear.astype(np.float)
        toa,r=sin_toa(doy.values[:],12.,27.99,0.)
        toa_idx=sw_day_max.index
    
    
    # Compute mean wind as f(hour)
#    ws_max.index=ws_max.index+datetime.timedelta(hours=6)
#    gust_hr=ws_max.groupby(ws_max.index.hour).median()
#    ax2.plot(gust_hr.index,gust_hr.values[:],color=cols[count])
    
    
    # Increment counter
    count +=1
    
# Tidy
xs=np.array([min_date,min_date,end_pre,end_mons,max_date])
ys=np.array([0.,40.,0.,40.])

ax.flat[0].grid("on"); ax.flat[0].set_xticklabels([]); \
                           ax.flat[0].set_xlim(min_date,max_date); \
                           ax.flat[0].set_ylabel("Temperature ($^{\circ}$C)");\
                           ax.flat[0].set_ylim(-30,25); \
                           ax.flat[0].axvspan(xs[0],xs[2],color='grey',alpha=0.25);\
                           ax.flat[0].axvspan(xs[3],xs[4],color='grey',alpha=0.25);\
                           ax.flat[0].set_xlim(min_date,max_date)
                           
ax.flat[1].grid("on"); ax.flat[1].set_xticklabels([]); \
                           ax.flat[1].set_xlim(min_date,max_date); \
                           ax.flat[1].set_ylabel("Wind speed (m/s)")
         

# Create reanlysis wind speed line
ref_winds=np.array(\
 [max_months.loc[max_months.index==ii].values[0] for ii in precip.index.month])   
ax.flat[1].plot(precip.index,ref_winds,color="grey",linestyle="--",linewidth=3)  
ax.flat[1].axvspan(xs[0],xs[2],color='grey',alpha=0.25)
ax.flat[1].axvspan(xs[3],xs[4],color='grey',alpha=0.25)                   
                        
ax.flat[2].set_ylabel("Relative Humidity (%)")                      
ax2=ax.flat[2].twinx(); ax2.plot(precip.index,precip.values[:],color="blue",linewidth=2); \
                        ax2.set_ylim(0,600);\
                        ax2.set_ylabel("Cumulative precip (mm)",color="blue");\
                        ax.flat[2].axvspan(xs[0],xs[2],color='grey',alpha=0.25);\
                        ax.flat[2].axvspan(xs[3],xs[4],color='grey',alpha=0.25); \
                        ax.flat[2].set_xlim(min_date,max_date); \
                        ax.flat[2].grid("on")
                        

ax.flat[3].grid("on"); ax.flat[3].set_xlim(min_date,max_date); \
                           ax.flat[3].set_ylabel("Max. insolation (W m$^{-2}$)");\
                           ax.flat[3].plot(toa_idx,toa,color='grey',\
                                  linestyle="--",linewidth=3); \
                        ax.flat[3].axvspan(xs[0],xs[2],color='grey',alpha=0.25);\
                        ax.flat[3].axvspan(xs[3],xs[4],color='grey',alpha=0.25);\
                        ax.flat[3].set_xlim(min_date,max_date)
#                        datetime.datetime(year=2019,month=7,day=8),color='grey',\
#                        alpha=0.1)
                        
# Plot the lapse rate
ax2=ax.flat[0].twinx()
ax2.plot(lapse.index,lapse.values[:]*1000.)
ax.flat[0].legend(ncol=2,frameon=True,loc=2,fontsize=8,fancybox=True,framealpha=0.5); \
ax2.set_ylabel("Lapse ($^{\circ}$C km$^{-1}$)",color="blue")
#ax.flat[2].set_xticks(t_day[idx][sel].index)
#ax.flat[2].xaxis.set_major_locator(mdates.DayLocator(interval=10)) 
#ax.flat[2].xaxis.set_minor_locator(mdates.DayLocator(interval=2))                              
fig.autofmt_xdate()
plt.subplots_adjust(right=0.8,hspace=0.03)
fig.savefig(dout+"AllObs.png",dpi=300)




