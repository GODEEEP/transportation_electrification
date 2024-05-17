import gcamreader
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import LDV
import sys

pev_dist='BEV'
class_dist='Equal'
pref_dist='Home60'
home_power_dist='MostL2'
work_power_dist='MostL2'

weather_scenarios=['rcp45cooler', 'rcp85hotter']

charging_strategies=[
    # home_access_dist, res_charging, work_charging
    ('HA75', 'min_delay', 'min_delay'),
    ('HA100', 'load_leveling', 'load_leveling'),
]

# monthly scaling factors to account for differing monthly travel behavior
monthly_travel_factors = np.array([234.1, 229.2, 269.5, 256.0, 280.3, 274.9, 279.3, 281.7, 273.1, 277.9, 257, 256.4])
monthly_travel_factors = monthly_travel_factors / monthly_travel_factors.mean()


def run_LDV_year_scenario(
    year,
    gcam_scenario,
    input_directory_path,
    api_key_path,
    weather_year=None,
    write_raw_loads=False,
):

    # get fleet size and daily miles breakdown by county
    fleet_size = LDV.get_fleet_by_county(
        f'{input_directory_path}/transportation_service_output_godeeep_{gcam_scenario}.csv',
        f'{input_directory_path}/transportation_assumptions_godeeep.xlsx',
        f'{input_directory_path}/UCD_techs_revised.csv',
        f'{input_directory_path}/EV_at_scale_2020_Kintner-Meyer_etal.csv',
        f'{input_directory_path}/county_timezones.dbx',
        year,
    )

    for home_access_dist, res_charging, work_charging in charging_strategies:

        cached_file = Path(f'./temperature_load_profiles_{gcam_scenario}_{year}_{home_access_dist}_{res_charging}.pickle')

        if cached_file.is_file():
            with open(cached_file, 'rb') as handle:
                temperature_load_profiles = pickle.load(handle)

        else:
            temperature_load_profiles = LDV.get_temperature_loads_by_county(
                fleet_size,
                'https://developer.nrel.gov/api/evi-pro-lite/v1/daily-load-profile',
                api_key_path,
                state_subset=['CA','OR','WA','AZ','NV','WY','ID','UT','NM','CO','MT'],
                pev_dist=pev_dist,
                class_dist=class_dist,
                pref_dist=pref_dist,
                home_access_dist=home_access_dist,
                home_power_dist=home_power_dist,
                work_power_dist=work_power_dist,
                res_charging=res_charging,
                work_charging=work_charging,
            )

            # save the profiles to a pickle file in case they are needed later (i.e. use it to skip the above cell)
            with open(cached_file, 'wb') as handle:
                pickle.dump(temperature_load_profiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for weather_scenario in weather_scenarios:

            loads = LDV.get_annual_hourly_load_profiles_by_county(
                fleet_size,
                temperature_load_profiles,
                f'{input_directory_path}/county_meteorology/{weather_scenario}/{weather_year if weather_year is not None else year}',
                weather_year if weather_year is not None else year,
            )

            fips = pd.read_csv(f'{input_directory_path}/EV_at_scale_2020_Kintner-Meyer_etal.csv')
            loads = loads.merge(fips[['State', 'FIPS']], how='left', on='FIPS')

            # scale by monthly factor
            loads['load_MWh'] = loads['load_MWh'] * pd.DataFrame(data={'month': loads.time.dt.month}).merge(
               pd.DataFrame(data={'month': np.arange(1,13), 'monthly_factor': monthly_travel_factors}),
               how='left',
               on='month'
            )['monthly_factor']

            # need to scale the EVI-Pro Lite load shapes to match with the GCAM state level output
            state_loads = loads.groupby('State')[['load_MWh']].sum()

            # read the GCAM transportation energy by state and convert to MW
            gcam_transportation_energy = pd.read_csv(
                f'{input_directory_path}/transportation_energy_output_godeeep_{gcam_scenario}.csv'
            )
            gcam_transportation_energy = gcam_transportation_energy[gcam_transportation_energy['Year'] == year]
            gcam_transportation_energy = gcam_transportation_energy[gcam_transportation_energy['technology'].isin(['BEV', 'Electric'])]
            # convert Exajoule to Megawatt
            gcam_transportation_energy['mw'] = gcam_transportation_energy['value'] * 277.77777777778 * 1000000 / 8760
            ldv_state_level_comparison = (gcam_transportation_energy[gcam_transportation_energy['subsector'].isin([
                '2W and 3W',
                'Car',
                'Large Car and Truck',
                'Light truck',
            ]) & gcam_transportation_energy.region.isin([
                'CA','OR','WA','AZ','NV','WY','ID','UT','NM','CO','MT'
            ])].groupby('region')[['mw']].sum() * 8760).merge(
                state_loads,
                left_index=True,
                right_index=True,
            ).rename(columns={'mw': 'GCAM_MWh', 'load_MWh': 'transportation_MWh'})

            ldv_state_level_comparison['scale_factor'] = ldv_state_level_comparison['GCAM_MWh'] / ldv_state_level_comparison['transportation_MWh']

            # merge the loads with the scale factors
            loads = loads.merge(
                ldv_state_level_comparison[['scale_factor']],
                left_on='State',
                right_index=True,
            )

            # scale the loads
            loads['load_MWh'] = loads['load_MWh'] * loads['scale_factor']

            if write_raw_loads:
                loads = loads.merge(
                    fleet_size[['FIPS', 'balancing_authority', 'fleet_size', 'daily_miles']],
                    how='left',
                    on=['FIPS', 'balancing_authority'],
                )
                loads.to_csv(
                    f"./{gcam_scenario}_{weather_scenario}_{res_charging}_{home_access_dist}_{year}_raw_LDV_county_loads.csv",
                    index=False
                )
                continue

            # aggregate the county loads to the balancing authority level
            ba_loads = LDV.aggregate_to_balancing_authority(loads)

            # write out csv files per balancing authority
            for group, data in ba_loads.groupby('balancing_authority'):
                output_path = Path(
                    f"./output/{gcam_scenario}/{weather_scenario}/{res_charging}/{group}_hourly_LDV_load_{gcam_scenario}_{weather_scenario}_{year}_{home_access_dist}_{res_charging}.csv"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_csv(output_path, index=False)




if __name__ == "__main__":

    year = int(sys.argv[1])
    scenario_name = sys.argv[2]
    input_directory = sys.argv[3]
    api_key_path = sys.argv[4]
    raw = False
    weather_year = None

    if len(sys.argv)==6:
        if len(sys.argv[5]) == 4:
            weather_year = int(sys.argv[5])
        elif sys.argv[5] == "1":
            raw = True

    run_LDV_year_scenario(
        year,
        scenario_name,
        input_directory,
        api_key_path,
        weather_year,
        raw,
    )



