# ------------------------------------------------------------------
# Simple script to visualize the NOAA remote sensing data
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

root_noaa = r'../NOAA_CORDEX/EUR-11'

# ------------------------------------------------------------------

# define the variables to be visualized
variables = ['VCI', 'TCI', 'VHI', 'mask_cold_surface']  # mask_cold_surface includes invalid pixels as well

# read the available years inside the noaa folder
years = os.listdir(root_noaa)
years.sort()

years = [year for year in years if not year.endswith('.nc') and not year.endswith('.json')]

# visualize each year separately
for year in years:
    # directory for the year
    dir_year = os.path.join(root_noaa, year)
    # read files inside the year directory
    files = os.listdir(dir_year)
    files.sort()

    # visualize each file for each year separately
    for file in files:
        # directory for the file
        dir_file = os.path.join(dir_year, file)
        # read the netcdf data
        data = xr.open_dataset(dir_file)

        # visualize each variable separately
        for v in variables:
            data[v].plot()
            plt.title('file=' + file[:-3])
            plt.show()
            plt.close()

