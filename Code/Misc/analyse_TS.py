#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script analyses everest time series (for trends/window analysis, etc)

More details to follow!
"""
import numpy as np, pandas as pd
import datetime
import matplotlib.pyplot as plt

# Name of text file with interpolated reanalysis data
fin="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/SouthCol.txt"

## Which month to extract
month=5

# Read in with pandas
data=pd.read_csv(fin,parse_dates=True,index_col=0,sep="\t")
#data=data.loc[data.index.month==5]

# Compute annual
ann=data.resample("A").mean()

fig,ax=plt.subplots(2,2)
fig.set_size_inches(10,4)
ax.flat[0].plot(ann["t"],marker=".",markersize=10,color="k")
ax.flat[1].plot(ann["ws"],marker=".",markersize=10,color="k")
ax.flat[2].plot(ann["rh"],marker=".",markersize=10,color="k")
ax.flat[3].plot(ann["p"],marker=".",markersize=10,color="k")


