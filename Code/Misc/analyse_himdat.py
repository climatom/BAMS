#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script analyses himdat data -- summit successes as function of time
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

#Read in
data=pd.read_csv\
("/home/lunet/gytm3/Everest2019/Research/BAMS/Data/Misc/himdata_data.csv",sep=',')

# Now try to parse dates in col msmtdate1
date_str=data["msmtdate1"]

# Convert to datetimes -- nothing this: 
# pandas.to_datetime(arg, errors='raise', dayfirst=False, \
# yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, #
# infer_datetime_format=False, origin='unix', cache=True)[source]
dates=[]
for i in date_str:
    try: dates.append(pd.datetime.strptime(i,"%d/%m/%Y"))
    except: continue
dates=pd.Series(np.ones(len(dates)),index=dates)

# Now figure out monthly freq
monthly=dates.groupby(dates.index.month).sum()/np.float(len(dates))*100.



