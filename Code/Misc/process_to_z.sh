#!/bin/bash

set -e

# Simple script to (a) compute geopotential on model levels; and 
# (b) interpolate met variables in dim z

# Set location of program(s)
exe="/home/lunet/gytm3/Everest2019/Research/Code/z_on_ml.py" # computes geopotential on model levels 
# Signature is: 
#
#Parameters    : tq.grib                - grib file with all the levelist
#                                         of t and q
#                zlnsp.grib             - grib file with levelist 1 for params
#                                         z and lnsp
#                -l levelist (optional) - slash '/' separated list of levelist
#                                         to store in the output
#                -o output   (optional) - name of the output file
#                                         (default='z_out.grib')
#"""

# Set filenames and options
# Directory holding all the tq/lnsp files
di="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/"

## Pattern of tq files
tqpat="Turb"

## Pattern of lnsp files
lnsppat="SpGp"

# Set lon/lat to interpolate to
lonlat="lon=86.9295_lat=27.9719" # <-- south col

#lonlat="lon=86.9250_lat=27.9881"# <-- summit loc

# Control script behaviour
a="True" # True = compute z on model levels
b="False" # True = interpolate to correct height

if [ ${a} == "True" ]; then

	count=0

	for tq in ${di}${tqpat}*; do # tq is turb file
		

		echo "Files are:"

		# zlnsp file
		zlnsp="${tq/$tqpat/$lnsppat}"
	        echo "   ${tq}"
                echo "   ${zlnsp}"


		# Call program -- output will be z_${count}.grib
		python ${exe} ${tq} ${zlnsp} -o ${di}scratch.grib

		# Interpolate tq, z and zlnsp to the same spot
                cdo -O -f nc remapbil,${lonlat} ${tq} ${di}interp_tq.nc # tq 
                cdo -O -f nc remapbil,${lonlat} ${di}scratch.grib ${di}interp_z.nc # z 
		cdo -O -f nc remapbil,${lonlat} ${zlnsp} ${di}interp_zlnsp_${count}.nc # lnsp 

                # oname -- increments by one each time
		# oname="/home/lunet/gytm3/Everest2019/Research/ReanalysisData/z_${count}.grib"

		# Merge z with the interpolated tq field 
		cdo -O -f nc merge ${di}interp_tq.nc ${di}interp_z.nc ${di}z_Turb_${count}.nc

		# Update!
		echo "Processed file: ${count}..."

		# Increment counter!
		count=$((count+1))

	done
fi

cdo -O -f nc mergetime ${di}z_Turb_*.nc ${di}merged_z_Turb.nc
cdo -O -f nc mergetime ${di}interp_zlnsp*.nc ${di}merged_zlnsp.nc
rm z*grib
# And remember to remove other files (interp*.nc, z_Turb_.nc)


