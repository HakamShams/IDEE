# ------------------------------------------------------------------
# Simple script to visualize the CERRA data
# ------------------------------------------------------------------

import xarray as xr
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

np.set_printoptions(suppress=True)
xr.set_options(display_max_rows=40)
xr.set_options(display_width=1000)

# ------------------------------------------------------------------

root_cerra = r'../CERRA'

# ------------------------------------------------------------------

# define the variables to be visualized
variables = ['al', 'hcc', 'lcc', 'liqvsm', 'mcc', 'msl', 'r2', 'si10', 'skt',
             'sot', 'sp', 'sr', 't2m', 'tcc', 'tciwv', 'tp', 'vsw', 'wdir10']

# read the available years inside the cerra folder
years = os.listdir(root_cerra)
years.sort()

years = [year for year in years if not year.endswith('.nc') and not year.endswith('.json')]

# visualize each year separately
for year in years:
    # directory for the year
    dir_year = os.path.join(root_cerra, year)
    # read files inside the year directory
    files = os.listdir(dir_year)
    files.sort()

    # visualize each file for each year separately
    for file in files:
        # directory for the file
        dir_file = os.path.join(dir_year, file)
        #read the netcdf data
        data = xr.open_dataset(dir_file)

        # visualize each variable separately
        for v in variables:
            data.sel(statistic='mean')[v].plot()
            plt.title('file=' + file[:-3])
            plt.show()
            plt.close()

