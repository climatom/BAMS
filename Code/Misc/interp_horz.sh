#!/bin/bash 
# Simple script to iterate over 4d vars and interpolate in lat and lon
# Ouput is a time-merged file 

set -e

# Directory
di="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/"

# Pattern to match 
inpat="SpGp"

# Final output name
outname="${di}merged_${inpat}.nc"

# Lon_Lat to interpolate to 
lonlat="lon=86.9250_lat=27.9881"

# Init counter for loop
count=1
for f in ${di}${inpat}*; do
        
	cmd="cdo remapbil,${lonlat} ${f} scratch_${inpat}_${count}.grb"
	${cmd}
        count=$((count+1))
done

# Here we merge all scratch files
cmd="cdo mergetime scratch_${inpat}*.nc ${outname}"
${cmd}

# Now we remove all the scratch files
cmd="rm scratch_${inpat}*.grib"
${cmd}




