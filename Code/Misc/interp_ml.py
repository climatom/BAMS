#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Python script to interpolate met data vertically from model levels.

Inputs: 
    
    1. Name of met file (also contains geopotential on model levels)
    2. Name of ln surface pressure / geopotential file
    3. Elevation to interpolate to (m)
    4. Name of output file
    
Note that inputs are identified by order, not by name!

Output: 
    
    1. Text file orgainsed with the following tab-delimited columns: 
        a.     | Date/time 
        b. --> | Interpolated met variables (read from the netcdf file) 
        
C. Tom Matthews, 09/07/2019
"""
import os, sys, numpy as np, netCDF4, datetime
import pandas as pd, GeneralFunctions as GF

def pf_ml(table_def,lnsp_file):
    
    """ 
    Computes the pressure at the full model level, given the grib
    table definition from ECMF, and the 1D (time) series of surface pressure
    """
    
    # Read in table def
    table=pd.read_csv(table_def,sep=",")
    nlevs=table.shape[0]-1
    
    # Read in zlnsp
    ncfile=netCDF4.Dataset(lnsp_file,"r")
    lnsp=np.squeeze(ncfile.variables["lnsp"][:])
    nt=lnsp.shape[0]
    
    pf=np.zeros((nt,nlevs))
    for t in range(nt):
        # Pressure on half levels 
        ph=table["a"].values[:]+table["b"].values[:]*np.exp(lnsp[t])
        # Pressure on full levels
        pf[t,:]=1/2.*(ph[0:-1]+ph[1:])/100. # From Pa to hPa
    
    return pf

# Constants
g=9.80665

# Get input args
mname=sys.argv[1]
lnsp_file=sys.argv[2]
elev=int(sys.argv[3])
ofile=sys.argv[4]

# Read the netcdf file in.
metf=netCDF4.Dataset(mname,"r")

# Note, z and met have dims nt x nlevs
# Convert z to m 
z=np.squeeze(metf.variables["z"][:]/g); 
print "Found z. Min: %.0fm, Max: %.0fm" % (np.min(z),np.max(z))
print "First 10 vals (time-step 1):"
print z[0,:10]
nt=z.shape[0]

# Get the time variable
ntime=metf.variables["time"]
n2d=netCDF4.num2date(ntime[:],units=ntime.units,calendar=ntime.calendar)


# Get dimensions so that we can exclude from vars
dims=[ii for ii in metf.dimensions]; 
print "________________________________________________________________"
print "Read file with dims:      "
for d in dims: 
	print "\t\t\t",d
print "& variables:              " 
# Get list of variables 
vs=[ii for ii in metf.variables if ii not in dims \
	      and "hy" not in ii and "z" not in ii]

for v in vs: 
	print "\t\t\t",v
print "________________________________________________________________"

print "\t * * * * Interpolating * * * * \n\n"
# Loop over vars
data={}
for v in vs:

    # Interpolate with numpy
    # Note: axes are reversed, otherwise result is nonsense with numpy
    data[v]=np.array([\
    np.interp(elev,z[ii,::-1],np.squeeze(metf.variables[v][ii,::-1])) \
       for ii in range(nt)])
    print "... interpolated %s to %.0fm" % (v,elev)
    
# Compute pressure on model levels
pf=pf_ml("/home/lunet/gytm3/Everest2019/Research/level60def.txt.csv",lnsp_file)
data["p"]=np.array([\
    np.interp(elev,z[ii,::-1],pf[ii,::-1]) for ii in range(nt)])
print "... interpolated p to %.0fm" % elev
print "\n * * * * Interpolation complete * * * * \n\n"

# Compute RH (0-100). Note Q2Vp returns hPa; satVp returns Pa
data["rh"]=GF.Q2vp(data["q"],data["p"])*100./GF.satVp(data["t"])*100.
data["rh"][data["rh"]>100]=100 # Limit to max = 100


# Check for T and adjust accordingly
if "t" in vs: data["t"]=data["t"]-273.15
# Ditto for wind speed
if "u" in vs and "v" in vs: data["ws"]=\
np.sqrt(np.power(data["u"],2)+np.power(data["v"],2))
# Update variables list
vs+=["ws"]; vs+=["p"]; vs+=["rh"]

# Ditto for q
if ["q"] in vs:
   data["q"]*=1000. # Changes units to g/kg
   

# Prepare header to write out 
header="\t".join(vs)
# Add date to header
header="date\t"+header+"\n"

with open(ofile,"w") as fo:

    # Header    
    fo.write(header)

    # Iterate over all lines and write  
    for ln in range(nt):
        
        # Valid time
        line_out="%02d/%02d/%02d %02d:%02d:00\t" % (n2d[ln].year,n2d[ln].month,\
                                       n2d[ln].day,n2d[ln].hour,n2d[ln].minute)
        # Met vars out -- add to date
        line_out+="\t".join(["%.6f"%data[vs[ii]][ln] for ii in range(len(vs))])+"\n"
        
        # Write it
        fo.write(line_out)
    
