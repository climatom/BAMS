#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Computes the enery balance. 

Notes that computation requires: 
    
    - Temp (in K)
    - Pressure (in Pa)
    - RH (fraction)
    - Ws (m/s)
    - Incident short/longwave radiation (W/m**2)
    - Reflected shortwave radiation (for QA purposes)
    
"""
# SEB code repository
import core # Library of SEB computation
import seb_utils # Library of helper functions (e.g. convert met vars)
# Public modules
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Local functions
# returns number of nans in an array
def count_nan(arrin):
    return np.sum(np.isnan(arrin))

# Options/Parameters
din="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
fs=["south_col_filled.csv","c2_filled.csv","summit_guess.csv"]
outs=["south_col","c2","summit"]
end_date=datetime.datetime(year=2019,month=11,day=1)
odir="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/SEB/"
odir_fig="/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/"

# Preallocation
out={}
input_data={}
        
# Params
z0_m = 13.*1e-4# Roughness length for snow (Stitger et al., 2019)
z0s = [z0_m/10.,z0_m,z0_m*10]

# Loop files 
for jj in range(len(fs)):
            
            # Update
            print "Computing SEB for: ",fs[jj]

            # Read in data
            fin=din+fs[jj]
            data=pd.read_csv(fin,sep=",",parse_dates=True, index_col=0,\
                             na_values="NAN")

            # Tuncate to ensure no more recent than 'end_date'
            data=data.loc[data.index<=end_date]
            
            # Resample to higher frequency --
            freq="%.0fmin" % (core.ds/60.)
            data=data.resample(freq).interpolate("linear")
                    
            # Assignments (for readability)
            ta=data["T_HMP"].values[:]+273.15
            ta2=data["T_109"].values[:]+273.15
            p=data["PRESS"].values[:]*100.
            rh=data["RH"].values[:]/100.      
            sw=data["SW_IN_AVG"].values[:].astype(np.float)
            sw_o=data["SW_OUT_AVG"].values[:].astype(np.float)
            lw=data["LW_IN_AVG"].values[:]
            
            # Wind speed names differ between files - deal with that here. 
            try:
                u=(data["WS_AVG"].values[:]+data["WS_AVG_2"].values[:])/2.
            except:
                u=data["WS_AVG"].values[:]
                
            # Met conversions
            vp=np.squeeze(np.array([seb_utils.SATVP(ta[i])*rh[i] for i in range(len(u))]))
            mr=seb_utils.MIX(p,vp)
            tv=seb_utils.VIRTUAL(ta,mr)
            qa=seb_utils.VP2Q(vp,p)
            rho=seb_utils.RHO(p,tv)
                                                           
            # Compute the SEB - with range of z0_m
            out[jj]=[]
            for zi in z0s:
                shf,lhf,swn,lwn,seb,ts,melt,sub,nit_log,qg = \
                core.SEB(ta,qa,rho,u,p,sw,lw,z0_m=zi)
                   
                # Store in DataFrame for easier computation/plotting 
                # indices 0: SCol; 1: Camp II; 2: Summit. For each 
                # out[jj], the energy balance is computed for a n-range
                # set of z0_m -- stored in a list
                scratch=pd.DataFrame({"Melt":melt,"Ablation":melt+sub,"T":ta,\
                              "shf":shf,"lhf":lhf,"sw":swn,"lw":lwn,\
                              "seb":seb,"qg":qg,"qmelt":melt*core.Lf/core.ds,\
                              "wind":u,"rh":rh,"ts":ts,"sub":sub},\
                              index=data.index)
                               
                # Write this out
                oname=odir+outs[jj]+"_z0_%.3f.csv"%(zi*1000)
                scratch.to_csv(oname,float_format="%.6f")
                # Summarise stats and write out, too (recycling oname)
                mu=scratch.mean()
                oname=odir+outs[jj]+"_mean_z0_%.3f.csv"%(zi*1000)
                mu.to_csv(oname,float_format="%.6f")
                std=scratch.resample("D").mean().std()
                oname=odir+outs[jj]+"_std_z0_%.3f.csv"%(zi*1000)
                std.to_csv(oname,float_format="%.6f")

                # store out in a list - for plotting
                out[jj].append(scratch)
 
                print("Computed SEB for z0_m = %.5fm"%zi)
    
            # Store input data in case we need to do something with it
            data["run_albedo"]=data["SW_OUT_AVG"].rolling(24*30).sum()/\
            data["SW_IN_AVG"].rolling(24*30).sum()
            input_data[fs[jj].replace(".csv","")]=data
            assert np.nanmax(data["run_albedo"]) <= 1.
         
            # Update
            print("Computed SEB for file: %s" %fin )

#=============================================================================#            
# Now plot 
#=============================================================================#

# Plot params 
y_lower=-10
y_upper=800
y_q_lower=-100
y_q_upper=100
xlower=np.min([out[ii][1].index[0] for ii in out])
xupper=np.max([out[ii][1].index[-1] for ii in out])
end_pre=datetime.datetime(year=2019,month=7,day=1)
end_mons=datetime.datetime(year=2019,month=10,day=1)
temp_lims=[-25,0]
temp_ticks=range(-25,5,5)
outdir="/home/lunet/gytm3/Everest2019/Research/BAMS/Figures/"
oname=outdir+"SEB_mod.png"

# Define energy components for plotting; and labels; and colours
coms=["shf","lhf","sw","lw","qg","qmelt"]
labels=["Q$_{H}$","Q$_{L}$","Q$_{SW}$","Q$_{LW}$","Q$_{G}$","Q$_{M}$"]
cols=["green","blue","orange","purple","grey","black"]

# Draw plots 
fig,ax=plt.subplots(2,2)
fig.set_size_inches(9,7)

###  C2 
# Note that out[x][0] is lower-bound z0; [x][1] is 'best guess'; and [x][2] 
# is upper-bound. 
# Plot melt and ablation: 
idx=out[1][1]["Melt"].index.to_pydatetime()
ax.flat[0].plot(idx,out[1][1]["Melt"].cumsum().values[:],color="k",label="Melt")
ax.flat[0].plot(idx,out[1][1]["sub"].cumsum().values[:],color="red",label="Sublimation")
ax.flat[0].fill_between(idx,out[1][0]["Melt"].cumsum().values[:],\
       out[1][2]["Melt"].cumsum().values[:],color="k",label="",alpha=0.3)
ax.flat[0].fill_between(idx,out[1][0]["sub"].cumsum().values[:],\
       out[1][2]["sub"].cumsum().values[:],color="red",label="",\
       alpha=0.3)
# Colour-in the monsoon
ax.flat[0].axvspan(xlower,end_pre,color='grey',alpha=0.25)
ax.flat[0].axvspan(end_mons,xupper,color='grey',alpha=0.25)
ax.flat[0].set_ylabel("Water Equivalent (mm)")
ax.flat[0].grid()
# Plot Temp -- creating first the T series for plotting -- 24*30 item moving average - 
# because we want daily mean and we have 2-min data
plotT=out[1][1]["T"].dropna().rolling(24*30).mean()-273.15; #print np.max(plotT)
ax2=ax.flat[0].twinx()
idx=plotT.index.to_pydatetime()
ax2.plot(idx,plotT.values[:],color='blue',alpha=0.5)
ax2.tick_params(axis='y',colors='blue')
ax2.set_ylim(temp_lims) 
ax2.set_yticks(temp_ticks)
ax2.set_yticklabels([])
# Energy comps
for i in range(len(coms)):
    scratch=out[1][1][coms[i]].resample("D").mean()
    n_nan=out[1][1][coms[i]].resample("D").apply(count_nan)
    scratch.loc[n_nan>6*30]=np.nan # Mask out any days with > 6 hours 
    # (# 25 %) missing data
    idx=scratch.index.to_pydatetime()
    idx=scratch.index.to_pydatetime()
    ax.flat[2].plot(idx[1:-1],scratch[1:-1],color=cols[i],label=labels[i])
# Colour-in the monsoon
ax.flat[1].axvspan(xlower,end_pre,color='grey',alpha=0.25)
ax.flat[1].axvspan(end_mons,xupper,color='grey',alpha=0.25)
    
###  South col
# Plot melt and ablation:
idx=out[0][1]["Melt"].index.to_pydatetime()
ax.flat[1].plot(idx,out[0][1]["Melt"].cumsum().values[:],color="k")
ax.flat[1].plot(idx,out[0][1]["sub"].cumsum().values[:],color="red",label="Ablation")
ax.flat[1].fill_between(idx,out[0][0]["Melt"].cumsum().values[:],\
       out[0][2]["Melt"].cumsum().values[:],color="k",label="",alpha=0.3)
ax.flat[1].fill_between(idx,out[0][0]["sub"].cumsum().values[:],\
       out[0][2]["sub"].cumsum().values[:],color="red",label="",\
       alpha=0.3)
ax.flat[1].grid()
# Plot Temp -- creating first the T series for plotting -- 24*30 item moving average - 
# because we want daily mean and we have 2-min data
plotT=out[0][1]["T"].dropna().rolling(24*30).mean()-273.15; #print np.max(plotT)
ax2=ax.flat[1].twinx()
idx=plotT.index.to_pydatetime()
ax2.plot(idx,plotT.values[:],color='blue',alpha=0.5)
ax2.tick_params(axis='y',colors='blue')
ax2.set_ylabel("Temperature ($^{\circ}$C)",color="blue")
ax2.set_ylim(temp_lims) 
ax2.set_yticks(temp_ticks)
# Colour-in the monsoon
ax.flat[2].axvspan(xlower,end_pre,color='grey',alpha=0.25)
ax.flat[2].axvspan(end_mons,xupper,color='grey',alpha=0.25)
# Energy comps
for i in range(len(coms)):
    scratch=out[0][1][coms[i]].resample("D").mean()
    # But have to hack-ishly find which days have more >5 (arbitary) NaNs and
    # set those dates to NaN
    n_nan=out[0][1][coms[i]].resample("D").apply(count_nan)
    scratch.loc[n_nan>6*30]=np.nan # Mask out any days with > 6 hours 
    # (# 25 %) missing data
    idx=scratch.index.to_pydatetime()
    ax.flat[3].plot(idx[1:-1],scratch[1:-1],color=cols[i],label=labels[i])
# Colour-in the monsoon
ax.flat[3].axvspan(xlower,end_pre,color='grey',alpha=0.25)
ax.flat[3].axvspan(end_mons,xupper,color='grey',alpha=0.25)
ax.flat[0].legend(loc=4)
ax.flat[2].legend(ncol=3,frameon=True)    
ax.flat[2].set_ylabel("Energy Flux (W m$^{-2}$)")
ax.flat[2].grid()
## Add in the 'summit 'melt/ablation series 
#idx=out[2][1]["Melt"].index.to_pydatetime()
#ax.flat[1].plot(idx,out[2][1]["Melt"].cumsum().values[:],color="k",linestyle='--')
#ax.flat[1].plot(idx,out[2][1]["sub"].cumsum().values[:],color="red",\
#       label="Ablation",linestyle='--')
#ax.flat[1].fill_between(idx,out[2][0]["Melt"].cumsum().values[:],\
#       out[2][2]["Melt"].cumsum().values[:],color="k",label="",alpha=0.3)
#ax.flat[1].fill_between(idx,out[2][0]["sub"].cumsum().values[:],\
#       out[2][2]["sub"].cumsum().values[:],color="red",label="",\
#       alpha=0.3)

# Edit all lims
ax.flat[0].set_ylim(y_lower,y_upper)
ax.flat[1].set_ylim(y_lower,y_upper)
ax.flat[2].set_ylim(y_q_lower,y_q_upper)
ax.flat[3].set_ylim(y_q_lower,y_q_upper)

ax.flat[0].set_xlim(xlower,xupper)
ax.flat[1].set_xlim(xlower,xupper)
ax.flat[2].set_xlim(xlower,xupper)
ax.flat[3].set_xlim(xlower,xupper)
ax.flat[3].set_xlim(xlower,xupper)
ax.flat[3].grid()
fig.autofmt_xdate()
# Save fig
fig.savefig(oname,dpi=300)

#=============================================================================#            
# Evaluation at the South Col -- using TS
#=============================================================================#

# Params
snow_thresh_lower=0.70
snow_thresh_upper=0.90
melt_thresh=core.boltz*core.emiss*273.15**4
dt=datetime.timedelta(days=1)
res="D"

test_mod=out[0][1]
test_obs=input_data["south_col_filled"]
scratch=test_obs.resample(res).mean()
scratch_mod=test_mod.resample(res).sum()  
scratch_obs=(test_obs.resample(res).max())["LW_OUT_MAX"]

fig,ax=plt.subplots(1,1)
fig.set_size_inches(8,4)
albedo_snow=np.zeros(len(scratch))*np.nan
albedo_no_snow=np.zeros(len(scratch))*np.nan
run_albedo=scratch["run_albedo"]
    
#Define indices:
# 1 -> snow on ground 
snow_idx=np.logical_and(run_albedo>=snow_thresh_lower,run_albedo<=snow_thresh_upper)
# 2 - > snow on ground and modelled melt 
mod_snow_melt=np.logical_and(snow_idx,scratch_mod["Melt"]>0)
# 3 - > no snow on ground and modelled melt
mod_nosnow_melt=np.logical_and(~snow_idx,scratch_mod["Melt"]>0)
# 4 -> snow on the ground and no modelled melt
mod_snow_nomelt=np.logical_and(snow_idx,scratch_mod["Melt"]<=0)
# 5 -> snow on ground and observed melt
obs_snow_melt=np.logical_and(snow_idx,scratch_obs>=melt_thresh)
# 6 -> no snow on the ground and oserved melt
obs_nosnow_melt=np.logical_and(~snow_idx,scratch_obs>=melt_thresh)
# 7 -> snow on the ground and no observed melt 
obs_snow_nomelt=np.logical_and(snow_idx,scratch_obs<melt_thresh)
      
# Now plot
# Albedo 
ax.plot(scratch["run_albedo"].index,scratch["run_albedo"],color="k")
# Modelled melt during "snow"
dummy=np.ones(len(scratch))*0.85
ax.scatter(scratch_mod.index[mod_snow_melt],dummy[mod_snow_melt],color="blue",s=5)
# Modelled during "no snow"
ax.scatter(scratch_mod.index[mod_nosnow_melt],dummy[mod_nosnow_melt],\
                color="blue",s=5,alpha=0.25)
# Observed during snow
ax.scatter(scratch_obs.index[obs_snow_melt],dummy[obs_snow_melt]*0.75,\
                color="red",s=5)
# Observed during no snow
ax.scatter(scratch_obs.index[obs_nosnow_melt],dummy[obs_nosnow_melt]*0.75,\
           color="red",s=5,alpha=0.25)
        
# Tidying
ax.fill_between(scratch_obs.index,np.zeros(len(scratch_obs)),\
                    np.ones(len(scratch_obs))*snow_thresh_lower,color='grey',alpha=0.25)
ax.fill_between(scratch_obs.index,np.ones(len(scratch_obs))*snow_thresh_upper,\
                    np.ones(len(scratch_obs))*1.12,color='grey',alpha=0.25)    
ax.set_xlim([scratch.index[0],scratch.index[-1]])
ax.set_ylim([0,1.1])
ax.grid()
fig.autofmt_xdate()
ax.set_ylabel("Albedo")
fig.savefig(odir_fig+"Albedo_melt_evidence.png",dpi=300)
    
# Now Summarise
# Number of hits:
hits=np.sum(np.logical_and(mod_snow_melt,obs_snow_melt))/\
            np.float(np.sum(mod_snow_melt))    
# Number of misses:
misses=np.sum(np.logical_and(mod_snow_nomelt,obs_snow_melt))/\
            np.float(np.sum(obs_snow_melt))
    
# Number of false positives:
false_alarms=1-hits
    
# Surface temperature series
mod_ts=np.zeros(len(scratch_obs))*np.nan
mod_ts[snow_idx]=test_mod.resample(res).mean()["ts"][snow_idx]-273.15
obs_ts=np.zeros(len(scratch_obs))*np.nan
scratch_obs=(test_obs.resample(res).mean())["LW_OUT_MAX"]
obs_ts[snow_idx]=[np.min([273.15,\
          (scratch_obs.loc[snow_idx][i]/(core.boltz*core.emiss))**0.25])-273.15\
            for i in range(np.sum(snow_idx)) ]
ts_series=pd.DataFrame(data={"obs":obs_ts,"mod":mod_ts},\
                           index=scratch_obs.index)
plt_idx=ts_series.index<=datetime.datetime(year=2019,month=9,day=30,hour=23)
plot_obs=ts_series.loc[plt_idx]["obs"]#
plot_mod=ts_series.loc[plt_idx]["mod"]
r=np.corrcoef(plot_obs[~np.isnan(plot_obs)],\
                       plot_mod[~np.isnan(plot_mod)])[0,1]
fig_ts2,ax_ts2=plt.subplots(1,1)
fig_ts2.set_size_inches(4,4)
ax_ts2.scatter(plot_obs,plot_mod,s=5,color="k")
dummy=np.linspace(-20,-5,100)
ax_ts2.plot(dummy,dummy,color="red")
ax_ts2.set_xlim(-20,-5)
ax_ts2.set_ylim(-20,-5)
ax_ts2.set_ylabel("Modelled $^{\circ}$C)")
ax_ts2.set_xlabel("Observed $^{\circ}$C)")
ax_ts2.text(-18,-8,"r = %.2f" % r)
fig_ts2.savefig(odir_fig+"Daily_TS_scatter.png",dpi=300)

# Print
print "\n\n\n"
print "Hit rate: %.2f%%" % (hits*100)
print "False alarms: %.2f%%" % (false_alarms*100)
print "Misses: %.2f%%" % (misses*100)


#=============================================================================#            
# Any Queries (for the write-up)
#=============================================================================#
scol=input_data["south_col_filled"]
scol_daily=scol.resample("D").mean()
scol_hourly=scol.resample("H").mean()
dateMaxHour=scol_hourly.index[scol_hourly["T_HMP"]==np.max(scol_hourly["T_HMP"])]
dateMaxDay=scol_daily.index[scol_daily["T_HMP"]==np.max(scol_daily["T_HMP"])]
print "Max S. Col daily-mean", np.max(scol_daily)
print "Max S. Col hourly-mean", np.max(scol.resample("H").mean())
print "Max S. Col hourly-mean occurred on", dateMaxHour
print "Max S. Col daily-mean occurred on", dateMaxDay
print "Total sublimation at at the South Col (middle roughness)",\
(out[0][1]["sub"].cumsum())[-1]
print "Total sublimation at Camp II (middle roughness)",\
(out[1][1]["sub"].cumsum())[-1]

