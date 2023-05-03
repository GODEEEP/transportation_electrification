import click
from datetime import date, datetime
from glob import glob
import holidays
import json
import matplotlib.pylab as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
from time import sleep
from tqdm import tqdm
from typing import Iterable, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")


# MegaJoules (MJ) in an ExaJoule (EJ)
EJ_TO_MJ = 1e12

# miles (mi) in a kilometer (km)
KM_TO_MI = 0.621371

# difference between kelvin and celsius
KELVIN_TO_CELSIUS = -273.15

# days in a year
DAYS_IN_A_YEAR = 365

# GCAM subsectors representing Light Duty Vehicles (LDV)
LDV_SUBSECTORS = [
    '2W and 3W',
    'Car',
    'Large Car and Truck',
    'Light truck',
]

# GCAM modes representing LDV
LDV_MODES = [
    'LDV_2W',
    'LDV_4W',
    'Truck',
]

# GCAM technologies representing electric vehicles
LDV_TECHNOLOGIES = [
    'BEV'
]

# Parameter values allowed in an EVI-Pro Lite API request
EVI_PRO_LITE_ALLOWED_PARAMETERS = {
    "fleet_size": [30000, 10000, 1000],  # also allows anything > 30,000
    "mean_dvmt": [45, 35, 25],
    "temp_c": [40, 30, 20, 10, 0, -10, -20],
    "pev_type": ['PHEV20', 'PHEV50', 'BEV100', 'BEV250'],  # this is not really a parameter?
    "pev_dist": ['BEV', 'PHEV', 'EQUAL'],
    "class_dist": ['Sedan', 'SUV', 'Equal'],
    "home_access_dist": ['HA100', 'HA75', 'HA50'],
    "home_power_dist": ['MostL1', 'MostL2', 'Equal'],
    "work_power_dist": ['MostL1', 'MostL2', 'Equal'],
    "pref_dist": ['Home60', 'Home80', 'Home100'],
    "res_charging": ['min_delay', 'max_delay', 'midnight_charge'],
    "work_charging": ['min_delay', 'max_delay'],
}

# EVI-Pro Lite API only allows 1000 requests per hour, so we need to allow time between each request to accommodate
EVI_PRO_LITE_API_REQUEST_DELAY = 5  # i.e. ceiling of 60 minutes/hr * 60 seconds/minute / 1000 requests/hr

# Mapping between the county timezone codes and their [daylight, standard] UTC offsets
# note that some counties have two zones, but we will only consider the first
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


# Method to find the closest allowed parameter value
def find_closest_parameter(parameter: str, value: Union[int, float]) -> Union[int, float]:
    if parameter not in EVI_PRO_LITE_ALLOWED_PARAMETERS:
        return value
    if parameter not in ['fleet_size', 'mean_dvmt', 'temp_c']:
        return value
    array = np.asarray(EVI_PRO_LITE_ALLOWED_PARAMETERS[parameter])
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# method to determine if a day is a weekend or holiday
def is_weekend_or_holiday(dt: pd.Timestamp) -> bool:
    # TODO use state specific holidays? hard to tell which are really observed
    federal_holidays = holidays.US(years=dt.year)
    if dt in federal_holidays:
        return True
    if dt.weekday() >= 5:
        return True
    return False


# method to determine if a day falls within Daylight Savings Time or not
def is_daylight_savings(dt: pd.Timestamp) -> bool:
    # ok so dst begins on the second sunday in march and ends on the first sunday in november... smh
    dst_start = date(pd.year, 3, 8)
    if dst_start.weekday() != 6:
        dst_start.replace(day=(8 + (6 - dst_start.weekday()) % 7))
    dst_end = date(pd.year, 11, 1)
    if dst_end.weekday() != 6:
        dst_end.replace(day=(1 + (6 - dst_end.weekday()) % 7))
    return dst_start <= dt.date() < dst_end


# method to convert one meteorology file to a dataframe with timestamp
# awkward signature avoids multiprocessing issues
def process_meteorology_file(path_and_counties: Tuple[Union[str, Path], Iterable[int]]):
    path = Path(path_and_counties[0])
    counties = path_and_counties[1]
    # assumes year, month, day, and hour are in filename separated by underscores
    y, m, d, h = path.stem.split('_')[:4]
    df = pd.read_csv(path)[['FIPS', 'T2']].rename(columns={'T2': 'temperature_celsius'})
    df = df[df['FIPS'].astype(int).isin(counties)]
    # convert kelvin to celsius
    df['temperature_celsius'] = df['temperature_celsius'] + KELVIN_TO_CELSIUS
    # add time column and reference by beginning of hour rather than end (so that the daily mean works correctly)
    df['time'] = pd.to_datetime(f'{y}-{m}-{d}T{h}:00:00Z') - pd.Timedelta(hours=1)
    return df


def get_fleet_by_county(
    gcam_output_path: Union[str, Path],
    gcam_transportation_assumptions_path: Union[str, Path],
    gcam_transportation_mapping_path: Union[str, Path],
    ev_by_county_path: Union[str, Path],
    county_timezone_path: Union[str, Path],
    year: int,
    technologies: List[str] = None,
) -> pd.DataFrame:
    """
    Downscales GCAM service output (passenger-km traveled per year) to
    county level fleet size, daily miles traveled, and utc offset
    :param gcam_output_path: path to a GCAM-USA output CSV
    :param gcam_transportation_assumptions_path: path to a GCAM-USA transportation assumptions XLSX
    :param gcam_transportation_mapping_path: path to a GCAM-USA vehicle size.class to subsector mapping CSV
    :param ev_by_county_path: path to an electric vehicle fleet percentage by county CSV
    :param county_timezone_path: path to a county to timezone mapping DBX
    :param states: list of states to process
    :param year: year of conern
    :return: dataframe of county level fleet size, daily miles traveled, and UTC offsets
    """
    
    if technologies is None:
        technologies = LDV_TECHNOLOGIES

    # read the files
    gcam_output = pd.read_csv(gcam_output_path)
    gcam_assumptions = pd.read_excel(gcam_transportation_assumptions_path)
    gcam_mapping = pd.read_csv(gcam_transportation_mapping_path, skiprows=7)
    ev_county_data = pd.read_csv(ev_by_county_path)
    county_timezone = pd.read_csv(
        county_timezone_path,
        sep='|',
        index_col=False,
        names=['STATE', 'ZONE', 'CWA', 'NAME', 'STATE_ZONE', 'COUNTY', 'FIPS', 'TIME_ZONE', 'FE_AREA', 'LAT', 'LON']
    )[['FIPS', 'TIME_ZONE']].rename(columns={'TIME_ZONE': 'timezone'}).groupby('FIPS').first()

    # get passenger-km / yr from the Energy Service output for the LDV subsectors
    pkm_per_yr = gcam_output[
        gcam_output['subsector'].isin(LDV_SUBSECTORS) &
        gcam_output['technology'].isin(technologies) &
        (gcam_output.Year == year)
    ][['region', 'subsector', 'value']].rename(
        columns={
            'region': 'state',
            'value': 'pkm_per_year',
        }
    ).groupby(['state', 'subsector']).sum().reset_index()
    # convert millions-pkm to pkm
    pkm_per_yr['pkm_per_year'] *= 1e6

    # get the passengers per vehicle per vehicle class from the assumptions
    passengers_per_vehicle = gcam_assumptions[
        (gcam_assumptions['mode'].isin(LDV_MODES)) &
        (gcam_assumptions.variable == 'load factor') &
        (gcam_assumptions.UCD_technology.isin(technologies))
    ][['size.class', str(year)]].rename(columns={str(year): 'passengers_per_vehicle', 'size.class': 'size_class'})

    # get the km traveled per year per vehicle class from the assumptions
    km_per_vehicle = gcam_assumptions[
        gcam_assumptions.variable == 'annual travel per vehicle'
    ][['size.class', str(year)]].rename(columns={
        str(year): 'km_per_vehicle',
        'size.class': 'size_class'
    })
    
    # repeat "Light Truck and SUV" as "Truck (0-2.7t)", such that it maps correctly
    km_per_vehicle = pd.concat([
        km_per_vehicle,
        km_per_vehicle[km_per_vehicle['size_class'] == 'Light Truck and SUV'].assign(size_class='Truck (0-2.7t)')
    ])    

    # get the relationships between GCAM vehicle size.classes and subsectors
    sector_mapping = gcam_mapping[
        gcam_mapping.UCD_technology.isin(technologies) &
        gcam_mapping.tranSubsector.isin(LDV_SUBSECTORS)
    ][['size.class', 'tranSubsector']].rename(columns={
        'tranSubsector': 'subsector',
        'size.class': 'size_class'
    })

    # merge all the above by subsector and class
    fleet_data = pkm_per_yr.merge(
        sector_mapping, on='subsector', how='left'
    ).merge(
        passengers_per_vehicle, on='size_class', how='left'
    ).merge(
        km_per_vehicle, on='size_class', how='left'
    )

    # group by state and subsector and take the mean by vehicle size.class
    fleet_data = fleet_data.groupby(['state', 'subsector']).aggregate({
        'size_class': lambda x: ','.join(x),
        'passengers_per_vehicle': 'mean',
        'km_per_vehicle': 'mean',
        'pkm_per_year': 'mean',
    })

    # calculate the fleet size
    fleet_data['fleet_size'] = (
        fleet_data['pkm_per_year'] / fleet_data['passengers_per_vehicle'] / fleet_data['km_per_vehicle']
    )
    # calculate the daily miles traveled
    fleet_data['daily_miles'] = fleet_data['km_per_vehicle'] * KM_TO_MI / DAYS_IN_A_YEAR

    # calculate daily miles traveled weighted by state fleet size
    fleet_data['daily_miles'] = fleet_data.daily_miles * fleet_data.fleet_size / fleet_data.groupby(
        'state').fleet_size.transform('sum')

    #  sum up fleet size and weighted share of daily miles traveled by state
    fleet_data = fleet_data.groupby('state').aggregate({
        'fleet_size': 'sum',
        'daily_miles': 'sum',
    }).reset_index()

    # rename some columns and merge with the electric vehicle share by county
    fleet_data = ev_county_data.rename(columns={
        'State': 'state',
        'County': 'county',
        'Balancing Authority': 'balancing_authority',
        'Fleet Percentage': 'fleet_percentage',
    }).merge(fleet_data, how='left', on='state')

    # calculate the county level fleet size
    fleet_data['fleet_size'] = fleet_data['fleet_size'] * fleet_data['fleet_percentage']

    # subset to the interesting columns
    fleet_data = fleet_data[['state', 'county', 'FIPS', 'balancing_authority', 'fleet_size', 'daily_miles']]

    # merge the timezone data and determine the offsets
    fleet_data = fleet_data.join(county_timezone, how='left', on='FIPS')
    fleet_data['utc_offset_daylight'] = fleet_data.timezone.str[0].map({
        k: v[0] for k, v in TIMEZONE_CODES.items()
    })
    fleet_data['utc_offset_standard'] = fleet_data.timezone.str[0].map({
        k: v[1] for k, v in TIMEZONE_CODES.items()
    })

    return fleet_data


def get_temperature_loads_by_county(
    county_fleet: pd.DataFrame,
    api_url: str,
    api_key_path: Union[str, Path],
    state_subset: Iterable[str] = None,
    county_subset: Iterable[Union[int, str]] = None,
    balancing_authority_subset: Iterable[str] = None,
    pev_dist='BEV',
    class_dist='Equal',
    pref_dist='Home60',
    home_access_dist='HA75',
    home_power_dist='MostL2',
    work_power_dist='MostL2',
    res_charging='min_delay',
    work_charging='min_delay',
) -> dict:

    # setup requests to automatically retry failed requests a few times, with a backoff
    session = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https', HTTPAdapter(max_retries=retries))

    # load the EVI-Pro Lite API key
    with open(api_key_path, 'r') as f:
        api_key = f.read().splitlines()[0]

    # if a subset is requested, reduce to that subset
    subset = county_fleet
    if state_subset is not None:
        subset = subset[subset['state'].isin(state_subset)]
    if county_subset is not None:
        subset = subset[
            subset['county'].isin(county_subset) |
            subset['FIPS'].isin([int(f) for f in county_subset if (
                not (not isinstance(f, (int, float, complex)) or isinstance(f, bool))
            ) or f.isdigit()])
        ]
    if balancing_authority_subset is not None:
        subset = subset[subset['balancing_authority'].isin(balancing_authority_subset)]

    # store the results in a nested dictionary with keys like:
    # fips -> balancing authority -> temperature -> 'weekday'/'weekend' -> timeseries (in 15 minute increments)
    load_profiles = {}

    print('Fetching all possible temperature loads for each county from NREL EVI Pro Lite...')
    
    with tqdm(total=len(subset.index) - 1) as pbar:
    
      # loop over each county/BA and get the load data for each possible temperature
      for _, row in subset.iterrows():

          fleet_size = int(np.round(row.fleet_size))
          if fleet_size == 0:
              # TODO should we also skip other small fleet size values to avoid rounding to 1000 ???
              continue
          fleet_size_closest = find_closest_parameter('fleet_size', fleet_size)
          fleet_size = fleet_size_closest if fleet_size < np.amax(
              EVI_PRO_LITE_ALLOWED_PARAMETERS['fleet_size']
          ) else fleet_size
          daily_miles_traveled = find_closest_parameter('mean_dvmt', row.daily_miles)

          for temperature in EVI_PRO_LITE_ALLOWED_PARAMETERS['temp_c']:
              try:
                  api_response = session.get(
                      api_url,
                      params=dict(
                          api_key=api_key,
                          temp_c=temperature,
                          fleet_size=fleet_size,
                          mean_dvmt=daily_miles_traveled,
                          pev_dist=pev_dist,
                          class_dist=class_dist,
                          pref_dist=pref_dist,
                          home_access_dist=home_access_dist,
                          home_power_dist=home_power_dist,
                          work_power_dist=work_power_dist,
                          res_charging=res_charging,
                          work_charging=work_charging,
                      ),
                      timeout=30
                  )
                  api_response = api_response.json()
              except:
                  print('Something went wrong... params:')
                  print(dict(
                      api_key=api_key,
                      temp_c=temperature,
                      fleet_size=fleet_size,
                      mean_dvmt=daily_miles_traveled,
                      pev_dist=pev_dist,
                      class_dist=class_dist,
                      pref_dist=pref_dist,
                      home_access_dist=home_access_dist,
                      home_power_dist=home_power_dist,
                      work_power_dist=work_power_dist,
                      res_charging=res_charging,
                      work_charging=work_charging,
                  ))
                  print(api_response.content)

              if 'errors' in api_response:
                  print(json.dumps(api_response, indent=2))
                  raise ValueError('. '.join(api_response['errors']))

              try:
                # sum the different types of load for each profile
                weekday_load_profile = np.sum(
                    np.array([np.array(p) for p in api_response['results']['weekday_load_profile'].values() if p is not None]), axis=0)
                weekend_load_profile = np.sum(
                    np.array([np.array(p) for p in api_response['results']['weekend_load_profile'].values() if p is not None]), axis=0)
              except:
                print(json.dumps(api_response, indent=2))
                raise ValueError('Something went wrong...')

              # add these profiles to our dictionary
              if row.FIPS not in load_profiles:
                  load_profiles[row.FIPS] = {}
              if row.balancing_authority not in load_profiles[row.FIPS]:
                  load_profiles[row.FIPS][row.balancing_authority] = {}
              load_profiles[row.FIPS][row.balancing_authority][temperature] = {
                  'weekday': weekday_load_profile,
                  'weekend': weekend_load_profile,
              }

              # delay a bit to stay under request limit
              sleep(EVI_PRO_LITE_API_REQUEST_DELAY)
              
          pbar.update(1)

    return load_profiles


def get_annual_hourly_load_profile_for_county(
    county_fleet: pd.DataFrame,
    county_meteorology: pd.DataFrame,
    ba_temperature_profiles: dict,
    year: int,
) -> pd.DataFrame:

    county_name = county_fleet.iloc[0].county

    # convert meteorology to local time
    # we assume it is always daylight time, to avoid discontinuities and ugly code
    # (supposedly DST becomes permanent in the USA in 2023)
    # (some counties do not celebrate DST; these counties use whatever offset they typically use)
    daylight_offset = county_fleet.iloc[0].utc_offset_daylight
    county_meteorology = county_meteorology.copy()
    county_meteorology.loc[:, 'time'] = county_meteorology.time + pd.Timedelta(hours=daylight_offset)

    # subselect full days only (to ease the merging process later)
    if county_meteorology.iloc[0].time.hour != 0:
        county_meteorology = county_meteorology[
            county_meteorology.time.dt.date > county_meteorology.iloc[0].time.date()
        ]
    if county_meteorology.iloc[-1].time.hour != 23:
        county_meteorology = county_meteorology[
            county_meteorology.time.dt.date < county_meteorology.iloc[-1].time.date()
        ]

    # resample to daily mean
    daily_county_meteorology = county_meteorology.groupby(county_meteorology.time.dt.date).mean().reset_index()

    # determine holiday status
    daily_county_meteorology['is_weekend_or_holiday'] = daily_county_meteorology.time.map(is_weekend_or_holiday)

    # build load time series per BA
    loads = []
    for ba in ba_temperature_profiles:
        load = []
        # build load time series
        for _, row in daily_county_meteorology.iterrows():
            temperature = find_closest_parameter('temp_c', row.temperature_celsius)
            load.append(ba_temperature_profiles[ba][temperature]['weekend' if row.is_weekend_or_holiday else 'weekday'])
        # convert to timeseries dataframe
        load = np.array(load).ravel()
        load = pd.DataFrame(
            {'load_MWh': load},
            index=pd.date_range(
                start=daily_county_meteorology.iloc[0].time,
                periods=len(load)+1,
                freq='15min',
                closed='right',
            )
        )
        # resample to hourly and convert from kWh to MWh
        load = load.resample('H', label='right', closed='right').mean() / 1000
        # convert back to UTC
        load.index = load.index - pd.Timedelta(hours=daylight_offset)
        county_meteorology.loc[:, 'time'] = county_meteorology.time - pd.Timedelta(hours=daylight_offset)
        # select only the year of interest
        load = load[
            (load.index > datetime(year, 1, 1, 0, 0, 0)) & (load.index <= datetime(year + 1, 1, 1, 0, 0, 0))
        ]
        # join with temperature
        load = load.tz_localize('utc').merge(
          county_meteorology.set_index('time')[['temperature_celsius']],
          how='left',
          left_index=True,
          right_index=True,
        )
        # add county and ba to frame
        load['FIPS'] = county_fleet.iloc[0].FIPS
        load['county'] = county_name
        load['balancing_authority'] = ba
        loads.append(load.reset_index().rename(columns={'index': 'time'}))

    return pd.concat(loads)


def get_annual_hourly_load_profiles_by_county(
    county_fleet: pd.DataFrame,
    county_temperature_loads: dict,
    meteorology_path: Union[str, Path],
    year: int,
) -> pd.DataFrame:

    # need the full year of meteorology plus a day on either side to account for time zone spread
    meteorology_files = sorted(
        glob(f'{meteorology_path}/*{year-1}_12_31_*.csv') +
        glob(f'{meteorology_path}/*{year}*.csv') +
        glob(f'{meteorology_path}/*{year+1}_01_01_*.csv')
    )

    # list of county FIPS for which we have profiles
    counties = list(county_temperature_loads.keys())

    if len(counties) == 0:
        return []

    # parallel processing
    with Pool() as pool:
        # read all the meteorology files and produce temperature timeseries for relevant counties
        meteorology = pd.concat(
            pool.imap_unordered(
                process_meteorology_file,
                ((fips, counties) for fips in meteorology_files),
                chunksize=367,
            )
        ).sort_values('time')
        # for each county/BA, build a timezone aware hourly load profile
        # note that we assume a single dominant timezone per county
        load_profiles = pool.starmap(
            get_annual_hourly_load_profile_for_county,
            [[
                county_fleet[county_fleet.FIPS == fips],
                meteorology[meteorology.FIPS == fips],
                county_temperature_loads[fips],
                year,
            ] for fips in counties]
        )

    return pd.concat(load_profiles).reset_index(drop=True)


def plot_county_loads(
    loads: pd.DataFrame,
    fips: Union[str, int],
    show: bool = True,
) -> Iterable[plt.Figure]:
    county_loads = loads[loads.FIPS == int(fips)]
    county_name = county_loads.iloc[0].county
    figs = []
    for ba in county_loads.balancing_authority.unique():
        load = county_loads[county_loads.balancing_authority == ba].set_index('time').sort_index()
        fig, ax = plt.subplots()
        min_load = load.load_MWh.resample('D').min()
        max_load = load.load_MWh.resample('D').max()
        ax = min_load.plot(
            color='tab:blue', alpha=0.25, ax=ax,
            title=f'LDV load for {county_name} ({int(fips)}) County for {ba} Balancing Authority')
        ax = max_load.rename('LDV Load (MWh)').plot(
            color='tab:blue', alpha=0.25, legend=True, ylabel='LDV Load (MWh)', xlabel='Time (UTC)', ax=ax
        )
        ax.fill_between(min_load.index, min_load, max_load, alpha=0.25, color='tab:blue')
        ax = load.temperature_celsius.rename(
            'Average Temperature (°C)'
        ).resample('D').mean().plot(
            secondary_y=True, mark_right=False, legend=True, color='tab:orange', ax=ax
        )
        ax.set_ylabel('Average Temperature (°C)')
        ax.left_ax.get_legend().legendHandles[0].set_alpha(0.5)
        ax.left_ax.get_legend().legendHandles[1].set_color('tab:orange')
        ax.left_ax.get_legend().legendHandles[1].set_alpha(1)
        fig.tight_layout()
        figs.append(fig)
    if show:
        for fig in figs:
            fig.show(block=False)
    return figs


def plot_balancing_authority_loads(
    ba_loads: pd.DataFrame,
    balancing_authority: str,
    show: bool = True,
) -> plt.Figure:
    ba_load = ba_loads[ba_loads.balancing_authority == balancing_authority].set_index('time').sort_index()
    fig, ax = plt.subplots()
    min_load = ba_load.load_MWh.resample('D').min()
    max_load = ba_load.load_MWh.resample('D').max()
    ax = min_load.plot(
        color='tab:blue', alpha=0.25, ax=ax,
        title=f'LDV load for {balancing_authority} Balancing Authority')
    ax = max_load.rename('LDV Load (MWh)').plot(
        color='tab:blue', alpha=0.25, legend=True, ylabel='LDV Load (MWh)', xlabel='Time (UTC)', ax=ax
    )
    ax.fill_between(min_load.index, min_load, max_load, alpha=0.25, color='tab:blue')
    ax.get_legend().legendHandles[0].set_alpha(0.5)
    fig.tight_layout()
    if show:
        fig.show(block=False)
    return fig


def aggregate_to_balancing_authority(
    county_loads: pd.DataFrame,
) -> pd.DataFrame:
    return county_loads[[
        'time', 'balancing_authority', 'load_MWh'
    ]].groupby(['time', 'balancing_authority']).sum().reset_index()


def write_to_csv(balancing_authority_loads: pd.DataFrame, year: int, output_path: Union[str, Path]):
    # create one timeseries file per balancing authority
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for group, data in balancing_authority_loads.groupby('balancing_authority'):
        data.to_csv(f"{output_path}/{group}_hourly_LDV_load_{year}.csv", index=False)
    pass


def downscale_ldv_load(
    gcam_output_path: Union[str, Path],
    gcam_transportation_assumptions_path: Union[str, Path],
    gcam_transportation_mapping_path: Union[str, Path],
    ev_by_county_path: Union[str, Path],
    county_timezone_path: Union[str, Path],
    meteorology_path: Union[str, Path],
    evi_pro_lite_api_key_path: Union[str, Path],
    evi_pro_lite_url: str,
    year: int,
    load_output_path: Union[str, Path],
    states: Iterable[str] = None,
    counties: Iterable[str] = None,
    balancing_authorities: Iterable[str] = None,
):
    county_fleet = get_fleet_by_county(
        gcam_output_path,
        gcam_transportation_assumptions_path,
        gcam_transportation_mapping_path,
        ev_by_county_path,
        county_timezone_path,
        year,
    )
    county_temperature_loads = get_temperature_loads_by_county(
        county_fleet,
        evi_pro_lite_url,
        evi_pro_lite_api_key_path,
        states,
        counties,
        balancing_authorities,
    )
    county_loads = get_annual_hourly_load_profiles_by_county(
        county_fleet,
        county_temperature_loads,
        meteorology_path,
        year,
    )
    balancing_authority_loads = aggregate_to_balancing_authority(county_loads)
    write_to_csv(balancing_authority_loads, year, load_output_path)
    return balancing_authority_loads


@click.command()
@click.option(
    '--gcam-output-path',
    default='./input/EMF_Results_US_states_0308.csv',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the GCAM output CSV file?',
    help="""Path to the GCAM output CSV file."""
)
@click.option(
    '--gcam-transportation-assumptions-path',
    default='./input/tran_assumption.xlsx',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the GCAM transportation assumptions XLSX file?',
    help="""Path to the GCAM transportation assumptions XLSX file."""
)
@click.option(
    '--gcam-transportation-mapping-path',
    default='./input/UCD_techs_revised.csv',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the GCAM transportation mapping CSV file?',
    help="""Path to the GCAM transportation mapping CSV file."""
)
@click.option(
    '--ev-by-county-path',
    default='./input/EV_at_scale_2020_Kintner-Meyer_etal.csv',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the EVs by county CSV file?',
    help="""Path to the EVs by county CSV file."""
)
@click.option(
    '--meteorology-path',
    default='./input/county_meteorology',
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the directory containing the meteorology files?',
    help="""Path to the directory containing meteorology files."""
)
@click.option(
    '--county-timezone-path',
    default='./input/county_timezones.dbx',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the county timezone DBX file?',
    help="""Path to the county timezone DBX file."""
)
@click.option(
    '--evi-pro-lite-api-key-path',
    default='./input/nrel-api-key',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to the NREL EVI-Pro Lite API key file?',
    help="""Path to the NREL EVI-Pro Lite API key file."""
)
@click.option(
    '--evi-pro-lite-url',
    default='https://developer.nrel.gov/api/evi-pro-lite/v1/daily-load-profile',
    type=str,
    required=True,
    prompt='What is the URL to the NREL EVI-Pro Lite API?',
    help="""URL for the NREL EVI-Pro Lite API."""
)
@click.option(
    '--year',
    default='2035',
    type=str,
    required=True,
    prompt='Year to consider?',
    help="""Year."""
)
@click.option(
    '--states',
    default='AZ,CA,CO,ID,MT,NE,NM,NV,OR,SD,TX,UT,WA,WY',
    type=str,
    required=False,
    prompt='Comma separated list of states to consider? (blank for all)',
    help="""Comma separated list of states; None for all."""
)
@click.option(
    '--counties',
    default='',
    type=str,
    required=False,
    prompt='Comma separated list of county names or FIPS codes to consider? (blank for all)',
    help="""Comma separated list of county names or FIPS codes; None for all."""
)
@click.option(
    '--balancing-authorities',
    default='',
    type=str,
    required=False,
    prompt='Comma separated list of balancing authorities to consider? (blank for all)',
    help="""Comma separated list of balancing authorities; None for all."""
)
@click.option(
    '--load-output-path',
    default='./output',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    required=True,
    prompt='What is the path to a directory in which to write load output files?',
    help="""Path to the load output directory."""
)
def _downscale_ldv_load(
    gcam_output_path,
    gcam_transportation_assumptions_path,
    gcam_transportation_mapping_path,
    ev_by_county_path,
    meteorology_path,
    county_timezone_path,
    evi_pro_lite_api_key_path,
    evi_pro_lite_url,
    year,
    states,
    counties,
    balancing_authorities,
    load_output_path,
):
    if states is None or states.strip() == '':
        states = None
    else:
        states = [s.strip() for s in states.split(',')]

    if counties is None or counties.strip() == '':
        counties = None
    else:
        counties = [c.strip() for c in counties.split(',')]

    if balancing_authorities is None or balancing_authorities.strip() == '':
        balancing_authorities = None
    else:
        balancing_authorities = [ba.strip() for ba in balancing_authorities.split(',')]

    downscale_ldv_load(
        gcam_output_path,
        gcam_transportation_assumptions_path,
        gcam_transportation_mapping_path,
        ev_by_county_path,
        county_timezone_path,
        meteorology_path,
        evi_pro_lite_api_key_path,
        evi_pro_lite_url,
        int(year),
        load_output_path,
        states,
        counties,
        balancing_authorities,
    )
  

if __name__ == "__main__":
    _downscale_ldv_load()
