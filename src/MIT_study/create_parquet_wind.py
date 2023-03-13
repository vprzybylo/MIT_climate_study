import glob
import xarray as xr
import pandas as pd

def main():
    dfs = []
    for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]:
        print(year)
        list_of_paths = glob.glob(f'/rdma/dgx-a100/NYSM/archive/nysm/netcdf/proc/{year}/*/*', recursive=True)
        df = xr.open_mfdataset(list_of_paths, parallel=True).to_dataframe()[['wspd_merge', 'wmax_merge']].reset_index()
        dfs.append(df)
    print('end of year loop')
    all_years_df = pd.concat(dfs)

    all_years_df.to_parquet('/home/vanessa/hulk/MIT_study/data/mesonet_wind_2015_2022.parquet.gzip',
                compression='gzip')  

if __name__ == 'main':
    main()