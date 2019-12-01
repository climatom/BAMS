#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Parses the ISD metadata
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

fin="/home/lunet/gytm3/Everest2019/Research/Code/AWS_meta/isd-history.txt"
data={}
lat=[]; lon=[]; elev=[]; yrst=[]; yrstp=[]
with open(fin,"r") as filein:
    for f in filein:
        if len(f.split()[3:])<7:
            continue 
        try:
            lati = float(f[57:64])
            loni = float(f[65:72])
            elevi=float(f[74:81])
            yrsti=float(f[82:86])
            yrstpi=float(f[91:95])
        except: continue
    
        if lati == 0 and loni == 0: 
            continue # skip because it seems this indicates 
        # dodgy meta-data (like 8,000m+station in Afghanistan -- and can't sanity
        # test when coords aren't listed properly)
        if elevi < -900: 
            continue
          
        lat.append(lati)
        lon.append(loni)
        elev.append(elevi)
        yrst.append(yrsti)
        yrstp.append(yrstpi)
 
data["lat"]=lat; data["lon"]=lon; data["elev"]=elev; data["yrst"]=yrst
data["yrstp"]=yrstp        
meta=pd.DataFrame(data)
current=meta.loc[meta["yrstp"]==2019]

# Visualize
meta["elev"].hist()
current["elev"].hist()

# Take out only the highest 
highall=meta.loc[meta["elev"]>=5000]
highnow=current.loc[current["elev"]>=5000]

meta.to_csv("/home/lunet/gytm3/Everest2019/Research/Code/AWS_meta/ISD_out.csv")