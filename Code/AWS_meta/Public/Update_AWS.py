
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program pulls down AWS data from Camp II, Soth Col, and Balcony weather
stations. It updates a logfile with the latest observations -- appending as we
go.

"""
import requests, numpy as np
import pandas as pd, datetime 
import os

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# Define file URLs
# Hourly 
kcc_url="http://earthpulse-raw.nationalgeographic.org"+\
".s3-website-us-east-1.amazonaws.com/PhortseKCC_hr.dat"
bc_url="http://earthpulse-raw.nationalgeographic.org."+\
"s3-website-us-east-1.amazonaws.com/BaseCamp_hr.dat"
c2_url="http://earthpulse-raw.nationalgeographic.org" +\
".s3-website-us-east-1.amazonaws.com/CampII_Hr.dat"
south_url="http://earthpulse-raw.nationalgeographic.org"+\
".s3-website-us-east-1.amazonaws.com/South%20Col_hr.dat"
bal_url="http://earthpulse-raw.nationalgeographic.org"+\
".s3-website-us-east-1.amazonaws.com/Summit_hr.dat"

# Daily & 10-min
south_url_day = "http://earthpulse-raw.nationalgeographic.org.s3-website"+\
"-us-east-1.amazonaws.com/South%20Col_day.dat"
south_url_10 ="http://earthpulse-raw.nationalgeographic.org.s3-website"+\
"-us-east-1.amazonaws.com/South%20Col_TenMin.dat"
bal_url_day = "http://earthpulse-raw.nationalgeographic.org.s3-website"+\
"-us-east-1.amazonaws.com/Summit_day.dat"
bal_url_10 = "http://earthpulse-raw.nationalgeographic.org.s3-website"+\
"-us-east-1.amazonaws.com/Summit_TenMin.dat"

# Put in list 
urls=[kcc_url,bc_url,c2_url,south_url,bal_url,south_url_day,south_url_10,\
      bal_url_day,bal_url_10]

# Define output names
dout="/home/lunet/gytm3/Everest2019/AWS/Logging/" 
out_kcc=dout+"kcc_log.csv"
out_bc=dout+"basecamp.csv"
out_c2=dout+"c2.csv"
out_south=dout+"south_col.csv"
out_bal=dout+"balcony.csv"
out_col_day=dout+"south_col_day.csv"
out_col_10=dout+"south_col_10.csv"
out_bal_day=dout+"balcony_day.csv"
out_bal_10=dout+"balcony_10.csv"
test=dout+"TEST.csv"

# Put in list
outs=[out_kcc,out_bc,out_c2,out_south,out_bal,out_col_day,out_col_10,out_bal_day,\
      out_bal_10]

# Define record start/stop
today=datetime.datetime.now()
st_kcc=datetime.datetime(year=2019,month=4,day=16,hour=1)
st_bc=datetime.datetime(year=2019,month=10,day=10,hour=6)
st_c2=datetime.datetime(year=2019,month=5,day=21,hour=7)
st_col=datetime.datetime(year=2019,month=5,day=22,hour=5)
st_bal=datetime.datetime(year=2019,month=5,day=23,hour=1)
# Put in list
starts=[st_kcc,st_bc,st_c2,st_col,st_bal,st_col,st_col,st_bal,st_bal]

# Define column headers (verbose!)
heads_kcc=["record#","T_HMP","PRT","RH","SURF_TEMP","SUB_TEMP","PRECIP_CODE",\
           "PRECIP","PRECIP_24","WS_AVG","WDIR","WS_MAX","SW_IN_SAMP",\
           "SW_IN_MAX","SW_IN_AVG","SW_OUT_SAMP","SW_OUT_MAX","SW_OUT_AVG",\
           "LW_IN_SAMP","LW_IN_MAX","LW_IN_AVG","LW_OUT_SAMP","LW_OUT_MAX",\
           "LW_OUT_AVG","NET","SR50","PRESS"]
heads_bc=["record#","T_HMP","RH","SYNOP","PRECIP","PRECIP_24","PRESS"]

heads_c2=["#record","T_HMP","T_109","RH","T_125","T_250","T_375","T_500",\
          "T_625","T_750","T_875","T_1000","WS_MAX","WS_AVG","WDIR",\
          "SW_IN_AVG","SW_OUT_AVG","LW_IN_AVG","LW_OUT_AVG","NET","PRESS_SAMP",\
          "PRESS_MAX","PRESS_MIN","PRESS","SNOW_DEPTH","SR50"]

heads_col=["record#","T_HMP","T_109","RH","WS_MAX","WS_AVG","WDIR","WS_MAX_2",\
           "WS_AVG_2","WDIR_2","SW_IN_SAMP","SW_IN_MAX","SW_IN_AVG",\
           "SW_OUT_SAMP","SW_OUT_MAX","SW_OUT_AVG","LW_IN_SAMP","LW_IN_MAX",\
           "LW_IN_AVG","LW_OUT_SAMP","LW_OUT_MAX","LW_OUT_AVG","NET","PRESS"]

heads_bal=["record#","T_HMP","T_109","RH","WS_MAX_1","WS_AVG_1","WDIR_1",\
           "WS_MAX_2","WS_AVG_2","WDIR_2","PRESS"]

heads_col_day=["record#","BattV_Min","AirTC_Max","AirTC_Min","AirTC_Avg",\
               "T109_C_Max","T109_C_Min","T109_C_Avg","RH_Max","RH_Min",\
               "RH_Avg","WS","WD","WS_Max","SR01Up","SR01Up_Max","SR01Up_Avg",\
               "SR01Dn","SR01Dn_Max","SR01Dn_Avg","IR01UpCo","IR01UpCo_Max",\
               "IR01UpCo_Avg","IR01DnCo","IR01DnCo_Max","IR01DnCo_Avg",\
               "NetTot","BP_mbar_Max","BP_mbar_Min","BP_mbar_Avg"]

heads_col_10=["record#","AirTC","T109_C","RH","WS_Max","WS","WD","WS_2_Max",\
              "WS_2","WD_2"]

heads_bal_day=["record#","BattV_Min","AirTC_Max","AirTC_Min","AirTC_Avg",\
               "T109_C_Max","T109_C_Min","T109_C_Avg","RH_Max","RH_Min",\
               "RH_Avg","WS","WD","WS_Max","BP_mbar_Min","BP_mbar_Avg"]

heads_bal_10=["record#","AirTC","T109_C","RH","WS_Max","WS","WD","WS_2_Max",\
              "WS_2","WD_2"]

# Put in list
headers=[heads_kcc,heads_bc,heads_c2,heads_col,heads_bal,heads_col_day,\
         heads_col_10,heads_bal_day,heads_bal_10]

## Get files
count=0
for u in urls:
    temp=requests.get(u)
    output = open("temp_out.csv", 'wb')
    output.write(temp.content)
    output.close()
    # Read in using Pandas
    temp_in=pd.read_csv("temp_out.csv",skiprows=3,index_col=0,parse_dates=True,\
                        date_parser=dateparse)
    
    # Fix the day date here
    if "day.dat" in u:
        temp_in.index=temp_in.index-datetime.timedelta(days=1)
#    if "south_col.csv" in outs[count]:
#        assert 1==2
    
    # Cut data from before we'd finished fiddling with the station
    idx=temp_in.index>=starts[count]
    temp_in=temp_in.loc[idx]
    
    # Read in the log file
    if os.path.isfile(outs[count]):
        
        # File exists, so append temp_in. First, though, id new measurements
        log=\
        pd.read_csv(outs[count],skiprows=3,index_col=0,parse_dates=True)
        log.columns=headers[count]
        idx=temp_in.index>log.index[-1]
        temp_in=temp_in[idx]
        temp_in.columns=headers[count]
        
        # Now concatenate
        assert temp_in.shape[1]==log.shape[1],"Shapes of DFs don't match, Tom!"
        log=log.append(temp_in)
    
    else: log=temp_in
    
    # Write files out -- note that we will definitely have "log"
    log.columns=headers[count]
    log.to_csv(outs[count],sep=",",float_format="%.3f")
#    if "south_col" in outs[count] and "day.dat" in u: 
#        log.to_csv(test,sep=",",float_format="%.3f")
#        assert 1==2
    # Increment counter
    count+=1







