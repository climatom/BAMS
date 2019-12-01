## Treatment

# South Col
_flagged = hourly mean T replaced with mean of 10-min samples on days when min BattV drops below 11.8 V. If all samples in hour are the same, NaN inserted
	   We also process air pressure, leaving hourly samples in place if different from previous hour; otherwise NaN
_filled = same as above, but filled with CS109 in place of NaN. Unchanging pressure filled with daily mean. Radiation measurements also corrected with: 
night SW values set to zero; SW less than out set to out / albedo; LW modelled when SW less than out 

# Balcony
_flagged = same as South Col (pressure only)
_filled = same as South Col (pressure only)

# c2
_filled = same as South_Col for radiation; missing pressure values interpolated when surrounded by valid measurements, otherwise back/forward-filled (as appropriate)
