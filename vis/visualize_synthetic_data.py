# ------------------------------------------------------------------
# Simple script to visualize the synthetic data
# ------------------------------------------------------------------

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

np.set_printoptions(suppress=True)
xr.set_options(display_max_rows=40)
xr.set_options(display_width=1000)

# ------------------------------------------------------------------

root_datacube = r'../Synthetic/synthetic_CERRA/datacube_synthetic_CERRA.nc'

# ------------------------------------------------------------------

# open the netcdf file
data = xr.open_dataset(root_datacube)
# slice the data to read the first 520 - 624 time steps
data = data.isel(time=slice(52*10, 52*12))
# load anomalies events
anomaly = data['anomaly_extreme'].values  # V, T, Y, X
# load extreme events
extreme = data['extreme'].values  # T, Y, X
# load dynamic variables
variables = ['var_01', 'var_02', 'var_03', 'var_04', 'var_05', 'var_06']
datacube_dynamic = data[variables].to_array().values.astype(np.float32)  # V, T, Y, X

for t in range(datacube_dynamic.shape[1]):

    fig, axs = plt.subplots(3, 6)

    for v in range(len(variables)):
        axs[0, v].imshow(datacube_dynamic[v, t, :, :])
        axs[0, v].set_title(variables[v])

        axs[1, v].imshow(anomaly[v, t, :, :])
        axs[1, v].set_title(variables[v] + ' anomaly')

    axs[2, 0].imshow(extreme[t, :, :] > 0)
    axs[2, 0].set_title(u'Δt$_{0}$ extreme')

    for ax in axs.flatten():
        ax.set_axis_off()

    fig.suptitle('timestep=' + str(t), y=0.96)
    plt.show()
    plt.close()
