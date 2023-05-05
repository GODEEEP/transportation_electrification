import os
import pandas as pd
import numpy as np
import datetime
from datetime import date, time
import matplotlib.pyplot as plt
import math
import itertools
from calendar import monthrange
import statistics
from statistics import mode


import sys
import depot_charging as dc

def generate_MHDV_load_profiles(
                                weight_immediate,
                                weight_delay,
                                weight_minpower,
                                kW_list_charger,
                                kW_list_weight,
                                kWh_per_mile_list_vehichle_class,
                                year,
                                GCAM_case,
                                fleet_size,
                                n_samples,
                                seed,
                                veh_type):
    
    #---Global Initialization---
    
    EJ_to_MWh                = 1e12*0.000277778
    mnth                     = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    wecc_states              = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
    
    
    def filter_data(df, start_times_delete, end_times_delete, start_times_cut, frac_start_times, end_times_cut, frac_end_times):

        selected_columns                 = df[["vid","class_id", "start_ts", "end_ts", "distance_total"]]
        df_filter                        = selected_columns.copy()

        ##---There are multiple mobility instances for a vehicle (rows with same "vid"). Thus, code below assigns---
        ##---unique ID to each instance----

        df_filter         = df_filter.reset_index(drop=True)
        df_filter['vid']  = df_filter.index

        #---'start_ts' and 'end_ts' in the vehicle mobility data created above are series.--- 
        ##---Thus, code below converts them into python datetime object---
        ##---Also, drops mobility instances with missing date or time or both.---

        df_filter[['start_d','start_t']]  = df_filter['start_ts'].str.split(expand=True) #split datetime into date and time for 'start_ts'
        df_filter[['end_d','end_t']]      = df_filter['end_ts'].str.split(expand=True) #split datetime into date and time for 'end_ts'.                                                                     #If there are no dates or times, it will return None

        df_filter                         = df_filter.dropna()  # Drop rows with missing date or time or both in 'start_ts' and 'end_ts'
        start_tss                         = pd.to_datetime(df_filter['start_d'], format='%Y/%m/%d')  + pd.to_timedelta(df_filter['start_t'])
        df_filter['start_ts']             = start_tss
        end_tss                           = pd.to_datetime(df_filter['end_d'],   format='%Y/%m/%d')  + pd.to_timedelta(df_filter['end_t'])
        df_filter['end_ts']               = end_tss

        df_filter                        = df_filter.drop(['start_d', 'start_t', 'end_d', 'end_t'], axis =1)
        df_filter['start_ts']            = pd.to_datetime(df_filter['start_ts'].dt.strftime("%Y-%m-%d %H:%M:%S")) 
        df_filter['end_ts']              = pd.to_datetime(df_filter['end_ts'].dt.strftime("%Y-%m-%d %H:%M:%S"))

        #---Flag vehicle mobilities that have different start and end dates and remove them---
        df_filter['flag'] = 0
        df_filter.loc[df_filter['start_ts'].dt.date != df_filter['end_ts'].dt.date, 'flag'] = 1
        df_filter          = df_filter[df_filter['flag'].isin([0])]

        #--- Bar plots for original fleet mobility data---
    #         print("Bar plots for start and end times of original fleetdna data")
    #         df_filter['start_ts'].groupby([df_filter['start_ts'].dt.hour]).count().plot(kind="bar", figsize =(4,2))
    #         plt.show()
    #         df_filter['end_ts'].groupby([df_filter['end_ts'].dt.hour]).count().plot(kind="bar", figsize =(4,2))
    #         plt.show()

        #---Remove vehicle mobilities based on 'start_ts' and 'end_ts'---
        df_filter = df_filter[~df_filter['start_ts'].dt.hour.isin(start_times_delete)]
        df_filter = df_filter[~df_filter['end_ts'].dt.hour.isin(end_times_delete)]
        df_filter = df_filter.reset_index(drop = True)

        #---Partially remove mobilities based on 'start_ts' and 'end_ts'---
        indexes_start_ts = df_filter[df_filter['start_ts'].dt.hour.isin(start_times_cut)].sample(frac = frac_start_times, random_state =                                 seed).index
        df_filter        = df_filter.drop(indexes_start_ts)
        indexes_end_ts   = df_filter[df_filter['end_ts'].dt.hour.isin(end_times_cut)].sample(frac = frac_end_times, random_state = seed).index
        df_filter        = df_filter.drop(indexes_end_ts)

         #--- Bar plots for modified fleet mobility data---
    #         print("Bar plots for start and end times of modified fleetdna data")
    #         df_filter['start_ts'].groupby([df_filter['start_ts'].dt.hour]).count().plot(kind="bar", figsize =(4,2))
    #         plt.show()
    #         df_filter['end_ts'].groupby([df_filter['end_ts'].dt.hour]).count().plot(kind="bar", figsize =(4,2))
    #         plt.show()

        return df_filter


    ##---The function "agg_60_min_load_profile" converts 1-second resoultion load profile to an hourly resolution---
    ##---This function is a modified NREL's code ---
    
    def agg_60_min_load_profile (load_profile_df):

        s_in_60min = 60*60 

        # prepare idx slices
        start_idxs = np.arange(0, len(load_profile_df), s_in_60min)
        end_idxs   = np.arange(s_in_60min, len(load_profile_df) + s_in_60min, s_in_60min)

        # generate list of avg kw over 60-min increments
        avg_60min_kw = [] #init
        for s_idx, e_idx in zip(start_idxs, end_idxs):
            avg_60min_kw.append(load_profile_df['power_kW'][s_idx:e_idx].mean())

        times = [] #init
        for hour in range(24):
                times.append(str(datetime.time(hour, 0, 0)))

        # create pd.DataFrame
        agg_60min_load_profile_df = pd.DataFrame({'time': times,
                                                  'avg_power_kw': avg_60min_kw})

        return agg_60min_load_profile_df


    ##---Vehicle operating data obtained from NREL's FleetDNA data (https://www.nrel.gov/transportation/fleettest-fleet-dna.html)---  
    ##---Filter FleetDNA data. The mobility instances are filtered to best guess the return-to-base vehicle operations---
    
    df_del_van               = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_delivery_vans.csv'))
    df_del_van_filtered      = filter_data(df_del_van, start_times_delete = [0, 1, 2, 3, 4], end_times_delete = [23], 
                                           start_times_cut = [], frac_start_times = 0, 
                                           end_times_cut   = [21, 22], frac_end_times   = 0.7)  # frac: what % to delete
    
    df_del_van_ldv           = df_del_van_filtered[df_del_van_filtered['class_id'].isin([1, 2])]
    df_del_van_mdv           = df_del_van_filtered[df_del_van_filtered['class_id'].isin([3, 4, 5, 6])]
    df_del_van_hdv           = df_del_van_filtered[df_del_van_filtered['class_id'].isin([7, 8])]
    
    df_del_tru               = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_delivery_trucks.csv'))
    df_del_tru_filtered      = filter_data(df_del_tru, start_times_delete = [], end_times_delete = [],
                                           start_times_cut = [], frac_start_times = 0, 
                                           end_times_cut   = [22,23], frac_end_times = 0.8)
    
    df_del_tru_ldv           = df_del_tru_filtered[df_del_tru_filtered['class_id'].isin([1, 2])]
    df_del_tru_mdv           = df_del_tru_filtered[df_del_tru_filtered['class_id'].isin([3, 4, 5, 6])]
    df_del_tru_hdv           = df_del_tru_filtered[df_del_tru_filtered['class_id'].isin([7, 8])]

    df_scul_bus              = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_school_buses.csv'))
    df_scul_bus_filtered     = filter_data(df_scul_bus, start_times_delete = [3, 4, 5], end_times_delete = [19, 20, 21, 22, 23],
                                          start_times_cut = [], frac_start_times = 0, 
                                          end_times_cut   = [], frac_end_times = 0) 
    
    df_scul_bus_ldv          = df_scul_bus_filtered[df_scul_bus_filtered['class_id'].isin([1, 2])]
    df_scul_bus_mdv          = df_scul_bus_filtered[df_scul_bus_filtered['class_id'].isin([3, 4, 5, 6])]
    df_scul_bus_hdv          = df_scul_bus_filtered[df_scul_bus_filtered['class_id'].isin([7, 8])]
    
    df_transit_bus           = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_transit_buses.csv'))
    df_transit_bus_filtered  = filter_data(df_transit_bus, start_times_delete = [0, 1, 2, 3, 4], end_times_delete = [],
                                          start_times_cut = [], frac_start_times = 0, 
                                          end_times_cut   = [22, 23], frac_end_times = 0.9)
    
    df_transit_bus_ldv       = df_transit_bus_filtered[df_transit_bus_filtered['class_id'].isin([1, 2])]
    df_transit_bus_mdv       = df_transit_bus_filtered[df_transit_bus_filtered['class_id'].isin([3, 4, 5, 6])]
    df_transit_bus_hdv       = df_transit_bus_filtered[df_transit_bus_filtered['class_id'].isin([7, 8])]
   
    df_bucket_tru            = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_bucket_trucks.csv'))
    df_bucket_tru_filtered   = filter_data(df_bucket_tru, start_times_delete = [], end_times_delete = [],
                                          start_times_cut = [0, 1], frac_start_times = 0.9, 
                                          end_times_cut   = [22, 23], frac_end_times = 0.9)
    
    df_bucket_tru_ldv        = df_bucket_tru_filtered[df_bucket_tru_filtered['class_id'].isin([1, 2])]
    df_bucket_tru_mdv        = df_bucket_tru_filtered[df_bucket_tru_filtered['class_id'].isin([3, 4, 5, 6])]
    df_bucket_tru_hdv        = df_bucket_tru_filtered[df_bucket_tru_filtered['class_id'].isin([7, 8])]
   
    df_ser_van               = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_service_vans.csv'))
    df_ser_van_filtered      = filter_data(df_ser_van, start_times_delete = [], end_times_delete = [],
                                          start_times_cut = [0], frac_start_times = 0.5, 
                                          end_times_cut   = [23], frac_end_times = 0.5)
    
    df_ser_van_ldv           = df_ser_van_filtered[df_ser_van_filtered['class_id'].isin([1, 2])]
    df_ser_van_mdv           = df_ser_van_filtered[df_ser_van_filtered['class_id'].isin([3, 4, 5, 6])]
    df_ser_van_hdv           = df_ser_van_filtered[df_ser_van_filtered['class_id'].isin([7, 8])] 

    df_tractors              = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_class_8_tractors.csv'))
    df_tractors_filtered     = filter_data(df_tractors, start_times_delete = [0,1,2,3,4,5], end_times_delete = [19,20,21,22,23],
                                          start_times_cut = [], frac_start_times = 0, 
                                          end_times_cut   = [], frac_end_times = 0)
    
    df_tractors_ldv          = df_tractors_filtered[df_tractors_filtered['class_id'].isin([1, 2])]
    df_tractors_mdv          = df_tractors_filtered[df_tractors_filtered['class_id'].isin([3, 4, 5, 6])]
    df_tractors_hdv          = df_tractors_filtered[df_tractors_filtered['class_id'].isin([7, 8])]
 
    df_refuse_tru            = pd.read_csv(os.path.join('..', 'input','data_for_fleet_dna_refuse_trucks.csv'))
    df_refuse_tru_filtered   = filter_data(df_refuse_tru, start_times_delete = [], end_times_delete = [],
                                          start_times_cut = [], frac_start_times = 0, 
                                          end_times_cut   = [], frac_end_times = 0) 
    
    df_refuse_tru_ldv        = df_refuse_tru_filtered[df_refuse_tru_filtered['class_id'].isin([1, 2])]
    df_refuse_tru_mdv        = df_refuse_tru_filtered[df_refuse_tru_filtered['class_id'].isin([3, 4, 5, 6])]
    df_refuse_tru_hdv        = df_refuse_tru_filtered[df_refuse_tru_filtered['class_id'].isin([7, 8])]
 
    #---- Aggregate vehicles into ldv, mdv, and hdv---
    
    df_ldv                   = pd.concat([df_del_van_ldv, df_del_tru_ldv, df_scul_bus_ldv, df_transit_bus_ldv, 
                                      df_bucket_tru_ldv, df_ser_van_ldv , df_tractors_ldv, df_refuse_tru_ldv])

    df_mdv                   = pd.concat([df_del_van_mdv, df_del_tru_mdv, df_scul_bus_mdv, df_transit_bus_mdv, 
                                      df_bucket_tru_mdv, df_ser_van_mdv , df_tractors_mdv, df_refuse_tru_mdv])

    df_hdv                   = pd.concat([df_del_van_hdv, df_del_tru_hdv, df_scul_bus_hdv, df_transit_bus_hdv, 
                                      df_bucket_tru_hdv, df_ser_van_hdv , df_tractors_hdv, df_refuse_tru_hdv])


    ##---Assign kWh per mile for each class of vehicle---
    
    df_ldv.loc[df_ldv['class_id'].isin([1]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[0]
    df_ldv.loc[df_ldv['class_id'].isin([2]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[1]

    df_mdv.loc[df_mdv['class_id'].isin([3]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[2]
    df_mdv.loc[df_mdv['class_id'].isin([4]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[3]
    df_mdv.loc[df_mdv['class_id'].isin([5]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[4]
    df_mdv.loc[df_mdv['class_id'].isin([6]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[5]

    df_hdv.loc[df_hdv['class_id'].isin([7]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[6]
    df_hdv.loc[df_hdv['class_id'].isin([8]), 'kwh_per_mile'] = kWh_per_mile_list_vehichle_class[7]


    ##---There are multiple mobility instances for a vehicle (rows with same "vid"). Thus, code below assigns---
    ##---each instances unique ID. It must be done here as well because "vid" can be same for different types of vehicles---

    df_to_work          =  locals()['df_'+ veh_type].copy()
    df_to_work          = df_to_work.reset_index(drop=True)
    df_to_work ['vid']  = df_to_work.index
   
    ##---Initialization for obtaining yearly charging load profiles---

    avg_charge_prof_month = []
    
    ##---Generate load profiles for each month in a loop---
    
    for month in range(12):
        df_to_work_month       = df_to_work.loc[df_to_work['start_ts'].dt.month.isin([month+1])] 
        df_to_work_month.reset_index(inplace = True)

        if df_to_work_month.empty:            
            df_to_work_month  =  df_to_work.loc[df_to_work['start_ts'].dt.month.isin([month])] 

        ##---Disaggregating 'df_to_work_month' such that it has three time slots in a day: midnight to dwell, on-shift, return to midnight--
        ##---Starting midnight is 00:00:00 and ending midnight is 23:59:00 for the computation ease---

        df_three_slots           = pd.DataFrame()   
        for i, row in df_to_work_month.iterrows():
            only_date = row.start_ts.date()
            t1        = datetime.datetime.combine(only_date, datetime.time(0,0)) #midnight--start of a day
            t2        = datetime.datetime.combine(only_date, datetime.time(23,59,59)) #midnight---end of a day 
            temp      = pd.DataFrame({'vid': [row.vid, row.vid, row.vid], 'class_id': [row.class_id, row.class_id, row.class_id],  
                                'start_ts': [t1, row.start_ts, row.end_ts], 'end_ts': [row.start_ts, row.end_ts, t2],
                               'distance_total': [0, row.distance_total, 0], 'kwh_per_mile': row.kwh_per_mile})
            df_three_slots   = pd.concat([ df_three_slots, temp])  

        df_three_slots.reset_index(inplace = True)

        ##---Renaming and formating 'df_three_slots' dataframe to match NREL's code's input---
        df_veh_schedules                  = pd.DataFrame()
        df_veh_schedules['veh_op_day_id'] = df_three_slots['vid']
        df_veh_schedules['start_time']    = df_three_slots['start_ts'].dt.strftime("%H:%M:%S")
        df_veh_schedules['end_time']      = df_three_slots['end_ts'].dt.strftime("%H:%M:%S")
        df_veh_schedules['total_time_s']  = ((df_three_slots['end_ts']- df_three_slots['start_ts']).abs().dt.total_seconds()).astype(int)
        df_veh_schedules['on_shift']      = np.where(df_three_slots['distance_total'] >0, 1, 0)
        df_veh_schedules['vmt']           = df_three_slots['distance_total']
        df_veh_schedules['kwh_per_mile']  = df_three_slots['kwh_per_mile']

        ##---'df_three_slots', df_veh_schedules' consit three time slots per 'veh_op_day_id'.---
        ##---Thus, code below creates a dataframe with two shifts: on_shift (moving) and off-Shit (dwell).---
        ##---This data format is necessary for "dc.generate_load_profiles" functions in "depot_charging.py".--- 
        
        df_vech_op_days_temp          = (df_veh_schedules.groupby(['veh_op_day_id', 'on_shift']).agg({'total_time_s':'sum', 'vmt':'mean',
                                                        'kwh_per_mile': 'mean'}).reset_index()).astype('float')

        df_vech_op_days_temp['veh_op_day_id'] = df_vech_op_days_temp['veh_op_day_id'].astype(int)
        df_vech_op_days_temp['on_shift']      = df_vech_op_days_temp['on_shift'].astype(int)


        ##---NREL's code also requires total ON shift time and total OFF shift time.--- 
        ##---Thus, creating "time_off_shift_s", "time_on_shift_s".---

        df_vech_op_days = pd.DataFrame(df_vech_op_days_temp.drop(columns='veh_op_day_id').values.reshape(len(df_vech_op_days_temp)//2, 8),                                 #reshape 2 adjacent column to rows                                                                             
                            columns=['shift_off', 'time_off_shift_s', 'vmt_off', 'kwh_per_mile_off', 'shift_on', 'time_on_shift_s', 'vmt',                               'kwh_per_mile'], # set new column names
                            index=df_vech_op_days_temp['veh_op_day_id'].drop_duplicates()).reset_index() 
                            # reset_index for a new column set index by veh_op_day_id and 
            
        df_vech_op_days.drop(columns   = ['shift_off', 'shift_on', 'vmt_off', 'kwh_per_mile_off'])
        df_vech_op_days                = df_vech_op_days[["veh_op_day_id", "time_off_shift_s", "time_on_shift_s", "vmt", "kwh_per_mile"]]
        df_vech_op_days['total_time']  = df_vech_op_days['time_on_shift_s'] + df_vech_op_days['time_off_shift_s']


        ##----Writing to .csv files---
        df_vech_op_days.to_csv(os.path.join('..', 'output', f'{veh_type}_veh_op_days.csv'), index=False)
        df_veh_schedules.to_csv(os.path.join('..', 'output', f'{veh_type}_veh_schedules.csv'), index = False)
        

        ##---Generating EV charging profiles using three charging strategies developed by NREL---
        ##---The charging strategies are called by modified 'dc.generate_load_profiles' function below.--- 
        ##---The charging strategies are modified in 'src>> 'depot_charging.py'--- 

        charge_immediate = dc.generate_load_profiles(
                                   fleet_size  = fleet_size,
                                   charge_strat='immediate',
                                   n_samples   = n_samples,
                                   kw_list     = kW_list_charger,
                                   kw_weights  = kW_list_weight,
                                   seed        = seed,
                                   veh_type     = veh_type)
        
        charge_delay = dc.generate_load_profiles(
                                   fleet_size   = fleet_size,
                                   charge_strat ='delayed',
                                   n_samples    = n_samples,
                                   kw_list      = kW_list_charger,
                                   kw_weights   = kW_list_weight,
                                   seed         = seed,
                                   veh_type     = veh_type)
        
        charge_minpower = dc.generate_load_profiles(
                                   fleet_size   = fleet_size,
                                   charge_strat ='min_power', 
                                   n_samples    = n_samples,
                                   kw_list      = kW_list_charger,
                                   kw_weights   = kW_list_weight,
                                   seed         = seed,
                                   veh_type     = veh_type)

        #---Aggregate the data obtained from charging strategies and create an hourly time series EV charging---
        #---load profile for a given year---
        #---The functions 'dc.generate_load_profiles' returns averaged daily profile across fleets with 1 second resolution.---
        #---The daily profiles are repeated for a month , i.e., all days in a month have same profile.---
        #---Then, all monthly profiles are concatenated to generate a yearly profile.---

        avg_charge_prof_samples =  weight_immediate*charge_immediate + weight_delay*charge_delay + weight_minpower*charge_minpower
        ## Create time indices and aggregate data to hourly resolution: NREL's code modified---
        times = []
        for hour in range(24):
            for minute in range(60):
                for second in range(60):
                    times.append(str(datetime.time(hour, minute, second)))

        avg_charge_prof_samples       = list(avg_charge_prof_samples)
        avg_charge_profile_df_daily   = pd.DataFrame({'time': times,
                                        'power_kW': [avg_charge_prof_samples[0]] + avg_charge_prof_samples})
        
        avg_charge_profile_df_daily   = agg_60_min_load_profile(avg_charge_profile_df_daily)

        # Create yearly EV charging profile
        avg_charge_prof_month.extend([list(avg_charge_profile_df_daily['avg_power_kw'])]*monthrange(year, month+1)[1])


    charge_prof_fleetdna = np.array(list(itertools.chain.from_iterable(avg_charge_prof_month))).flatten()
    normalized_profile   = np.array(charge_prof_fleetdna/np.sum(charge_prof_fleetdna))


    ##---Adjust the normalized charging profiles to annual transportation energy in WECC (Exa Joules) projected by GCAM---

   
    energy_by_mode_and_fuel = pd.read_csv(os.path.join('..', 'input', 'transportation_energy_output_godeeep_' + GCAM_case +'.csv'))

    if veh_type == 'mdv':

        energy_ev = energy_by_mode_and_fuel[(energy_by_mode_and_fuel['input'] == 'elect_td_trn')
                    & energy_by_mode_and_fuel['subsector'].isin(['Medium truck'])]

    if veh_type == 'hdv':
        energy_ev = energy_by_mode_and_fuel[(energy_by_mode_and_fuel['input'] == 'elect_td_trn')
                    & energy_by_mode_and_fuel['subsector'].isin(['Bus', 'Heavy truck'])]

    energy_ev_wecc          = energy_ev[energy_ev['region'].isin(wecc_states)].reset_index(drop=True)
    energy_ev_wecc_year     = energy_ev_wecc[energy_ev_wecc['Year'].isin([year])].reset_index(drop=True)
    MW_ev_wecc              = normalized_profile*(energy_ev_wecc_year['value'].sum()*EJ_to_MWh)


    ##---Downscale the WECC EV charging profile to Balancing Authority (BA) level and adjust the profiles to UTC---
    ##---It is assumed that profiles obtained using FleetDNA data is in local time---

    # Mapping between the county timezone codes and their [daylight, standard] UTC offsets
    # note that some counties BAs have multiple timezones, we consider the most occured timezones
    # note that later we are further simplifying by assuming it is always daylight time
    # (supposedly this will be true from 2023 onward in the USA)

    TIMEZONE_CODES: dict = {
         'V': [-3, -4],    # Atlantic Standard
         'E': [-4, -5],    # Eastern Standard
         'C': [-5, -6],    # Central Standard
         'M': [-6, -7],    # Mountain Standard (m = daylight time not observed)
         'm': [-7, -7],
         'P': [-7, -8],    # Pacific Standard
         'A': [-8, -9],    # Alaska Standard
         'H': [-10, -10],  # Hawaii-Aleutian Standard (h = daylight time observed)
         'h': [-9, -10],
         'G': [10, 10],    # Guam & Marianas
         'J': [9, 9],      # Japan Time
         'S': [13, 13],    # Samoa Standard
    }

    #---Percentage of PEV penetration across BAs within WECC---

    df_ba_in_wecc_pev_percent = pd.read_csv(os.path.join('..', 'input','mdv_ba_percent_wecc.csv'))
    ba_in_wecc_pev_percent    = df_ba_in_wecc_pev_percent.set_index('Balancing Authority').to_dict()['MDV Penetration']

    MW_ev_wecc_2_more_days  = np.concatenate([MW_ev_wecc[0:24], MW_ev_wecc, MW_ev_wecc[8760-24:8760]] )
    #adding 2 days data to give enough data for UTC adjustments.                                                                                                                         

    time_series             = pd.Series(pd.date_range(start=f'{year-1}-12-31 1:00:00', end=f'{year+1}-1-2 00:00:00', freq='H')) #adjust time series
    time_series_orig        = pd.Series(pd.date_range(start=f'{year}-1-1 1:00:00', end=f'{year+1}-1-1 00:00:00', freq='H'))



    if len(ba_in_wecc_pev_percent.items())%2 == 0:
        fig1, axs1 = plt.subplots(nrows= len(ba_in_wecc_pev_percent.items())//2, ncols=2, figsize=(20,50)) # for plotting BA profiles
    else:
        fig1, axs1 = plt.subplots(nrows= len(ba_in_wecc_pev_percent.items())//2 +1, ncols=2, figsize=(20,50)) # for plotting BA profiles


    plt.subplots_adjust(hspace=0.7)
    cnt      = 1

    for ba, percent in ba_in_wecc_pev_percent.items():

        MW_ev_ba            = percent*MW_ev_wecc_2_more_days
        df_MW_ev_ba         = pd.DataFrame({'time': time_series, 'MW':MW_ev_ba, 'BA': ba,})

        #---Map FIPS to BA time zones---

        fips_timezone_code  = (pd.read_csv(os.path.join('..', 'input','county_timezones.dbx'),  sep='|',
            index_col=False,  names=['STATE', 'ZONE', 'CWA', 'NAME', 'STATE_ZONE', 'COUNTY',
                                     'FIPS', 'TIME_ZONE', 'FE_AREA', 'LAT', 'LON'])
           [['FIPS', 'TIME_ZONE']].rename(columns={'TIME_ZONE': 'timezone'}).groupby('FIPS').first()).reset_index()

        ba_fips             = pd.read_csv(os.path.join('..', 'input', 'EV_at_scale_2020_Kintner-Meyer_etal.csv'))
        ba_fips_list        = list(ba_fips['FIPS'][ba_fips['Balancing Authority'].isin([ba])])

        #---Some BAs have multiple time zones. Assign the most frequent timezone to such BAs.---

        timezone_code_ba    = mode(list(fips_timezone_code['timezone'][fips_timezone_code['FIPS'].isin(ba_fips_list)]))

        daylight_offset     = TIMEZONE_CODES[timezone_code_ba][0]

        df_MW_ev_ba['time'] = pd.to_datetime(df_MW_ev_ba['time'], format='%Y-%m-%d %H:%M:%S') - pd.Timedelta(hours=daylight_offset)

        df_MW_ev_ba.to_csv(os.path.join('..', 'output', f'{veh_type}_{GCAM_case}_{ba}_profile_{year}.csv'), index=False)

        #---BA-level plots in loop----
        if len(ba_in_wecc_pev_percent.items())%2 == 0:
            ax1 = plt.subplot(len(ba_in_wecc_pev_percent.items())//2, 2, cnt)
        else:
            ax1 = plt.subplot(len(ba_in_wecc_pev_percent.items())//2 +1, 2, cnt)
        ax1.plot(range(len(df_MW_ev_ba)), df_MW_ev_ba['MW'])
        ax1.set_xticks(np.linspace(0,len(df_MW_ev_ba), 12))
        ax1.set_xticklabels(mnth, fontsize=8)
        ax1.margins(x=0)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.ylabel('Power, MW', fontsize=16)
        plt.xlabel('Month of a year', fontsize=16)
        plt.title('{} Profile for {} {}'.format(veh_type, ba, year), fontsize=16 )
        plt.grid(axis ='both', linestyle='--')
        cnt = cnt+1

    plot_fp = os.path.join('..', 'output', f'{veh_type}_{GCAM_case}_ba_profile_{year}.png')
    plt.savefig(plot_fp, bbox_inches='tight', dpi=300)

    df_prof_normalized      = pd.DataFrame({'time': time_series_orig, 'kW_normalized': normalized_profile})
    df_prof_wecc            = pd.DataFrame({'time': time_series_orig, 'MW_WECC': MW_ev_wecc})


    df_prof_normalized.to_csv(os.path.join('..', 'output', f'{veh_type}_normalized_profile.csv'), index = False)


    ##---Normalized charging plot----
    fig2, ax1 = plt.subplots(figsize=(16,8))
    ax1.plot(range(len(normalized_profile)), normalized_profile)
    ax1.set_xticks(np.linspace(0,len(normalized_profile), 12))
    ax1.set_xticklabels(mnth, fontsize=8)
    ax1.margins(x=0)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel('Charging Fraction', fontsize=16)
    plt.xlabel('Month of a year', fontsize=16)
    plt.title('{} Normalized Profile Based on FleetFNA Data {}'.format((veh_type).upper(), year), fontsize=16 )
    plt.xlim(0, 8760)
    plt.grid(axis='both', linestyle='--')
    plot_fp = os.path.join('..', 'output', f'{veh_type}_normalized_profile.png')
    plt.savefig(plot_fp, bbox_inches ='tight', dpi=300)


