#!/bin/bash

set -e

# Simple script to in interpolate radiation fluxes to lon/lat position


# Set filenames and options
# Directory holding all the radiation files
di="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/"

# Turb file to merge with (we'll use Python for that...)

## Pattern of Rad files
radpat="Rad_ERA-I_"

# Set lon/lat to interpolate to
lonlat="lon=86.9295_lat=27.9719" # <-- south col

#lonlat="lon=86.9250_lat=27.9881"# <-- summit loc

count=1
for rad in ${di}${radpat}*; do # rad is radiation file

        # oname -- increments by one each time
        oname="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/r_${count}.nc"

	# Interpolate rad
        cdo -s -O -f nc remapbil,${lonlat} ${rad} ${oname} 

        # Update!
        echo "Processed file: ${count}..."

	# Increment counter!
	count=$((count+1))

done

# Merge all the individual files 
cdo -O -f nc -b 64 mergetime ${di}r_*.nc ${di}merged_rad.nc

# Extract tisr (ToA) and write to text file (with hour)
cdo outputtab,name,date,time,value, -selvar,tisr merged_rad.nc >SouthCol_tisr_raw.txt

# Open with python, process to hourly-mean values, and combine with the turb file(s) 

rm ${di}r_*.nc


