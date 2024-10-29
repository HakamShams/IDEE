# ------------------------------------------------------------------
"""
# Dataset class for Synthetic data

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import numpy as np
import xarray as xr
import os
import json
import torch
from torch.utils.data import Dataset
import warnings
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# ------------------------------------------------------------------


class Synthetic_Dataset(Dataset):
    """
        Dataset class for Synthetic dataset

        Attributes
        ----------
        root_datacube (str): directory to Synthetic dataset
        times (tuple): begin and ends of the time steps
        variables (list): list of synthetic dynamic variable names
        variables_static (list): list of synthetic static variable names
        delta_t (int): temporal resolution of the data
        is_aug (bool): option to use data augmentation
        is_clima_scale (bool): option to normalize the data with climatology
        is_norm (bool): option to normalize the data
        window_size (int): window to reduce resolution
        is_replace_anomaly (bool): option to replace anomalous events with normal data
        x_min (int): minimum longitude
        x_max (int): maximum longitude
        y_min (int): minimum latitude
        y_max (int): maximum latitude
        n_lat (int): number of pixels in the latitude/y direction
        n_lon (int): number of pixels in the longitude/x direction
        n_lat_window (int): number of pixels in the latitude/y direction after reducing the resolution
        n_lon_window (int): number of pixels in the longitude/x direction after reducing the resolution
        var_n_dynamic (int): number of dynamic variables
        var_n_static (int): number of static variables
        var_n (int): number of dynamic variables

        Methods
        -------
        __get_var_n()
            private method to get the number of variables
        __getpath()
            private method to get the files of the dataset inside the root directory

        __load_artificial_statistic()
            private method to get the statistics of the synthetic dataset from the root directory
        __load_artificial_climatology_pixels()
            private method to load synthetic climatology from the root directory
        __get_datacube()
            Private method to get the dataset inside the root directory

        min_max_scale(array, min_alt, max_alt, min_new, max_new)
            helper method to normalize an array between new minimum and maximum values

        __getitem__(index)
            method to load datacube by the file index
        __len__()
            method to get the number of files

        properties
        ----------
        datacube_dynamic()
            get the datacube dynamic data

        extreme()
            get the datacube extreme events data

        anomaly()
            get the datacube anomalous events data

        timestep()
            get the datacube time steps of the data
    """

    def __init__(self, root_datacube: str, times: tuple, variables: list, variables_static: list,
                 delta_t: int = 4, is_aug: bool = False,
                 is_clima_scale: bool = False, is_norm: bool = True, window_size: int = 1,
                 is_replace_anomaly: bool = False,
                 x_min: int = 0, x_max: int = 200,
                 y_min: int = 0, y_max: int = 200):
        """
        Args:
        ----------
        root_datacube (str): directory to Synthetic dataset
        times (tuple): begin and ends of the time steps
        variables (list, optional): list of synthetic dynamic variable names
        variables_static (list, optional): list of synthetic static variable names
        delta_t (int, optional): temporal resolution of the data. Default to 4
        is_aug (bool, optional): option to use data augmentation. Defaults to False
        is_clima_scale (bool, optional): option to normalize the data with climatology. Defaults to False
        is_norm (bool, optional): option to normalize the data. Defaults to True
        window_size (int, optional): window to reduce resolution. Defaults to 1
        is_replace_anomaly (bool, optional): option to replace anomalous events with normal data. Defaults to False
        x_min (int, optional): minimum longitude. Defaults to 0
        x_max (int, optional): maximum longitude. Defaults to 200
        y_min (int, optional): minimum latitude. Defaults to 0
        y_max (int, optional): maximum latitude. Defaults to 200
        """

        super().__init__()

        self.root_datacube = root_datacube
        self.exp_name = os.path.basename(os.path.normpath(root_datacube))

        self.delta_t = delta_t
        self.is_aug = is_aug
        self.is_shuffle = is_shuffle
        self.is_norm = is_norm
        self.is_clima_scale = is_clima_scale
        self.variables_dynamic = variables
        self.variables_static = variables_static
        self.times = times
        self.is_replace_anomaly = is_replace_anomaly

        self.window_size = window_size
        self.is_reduce = (window_size > 1)

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.n_lat = y_max - y_min
        self.n_lon = x_max - x_min

        self.n_lat_window = self.n_lat // window_size
        self.n_lon_window = self.n_lon // window_size

        # TODO add check for input

        # sort variables and years
        self.variables_dynamic.sort()
        self.variables_static.sort()

        self.__get_var_n()

        # preprocessing for the dataset
        if is_clima_scale or is_replace_anomaly:
            self.__load_artificial_climatology_pixels()
            self.__load_artificial_statistic()
        else:
            if is_norm:
                self.__load_artificial_statistic()

        self.__get_datacube()

    def __get_var_n(self):
        """
        Private method to get the number of variables
        """
        self.var_n_dynamic = len(self.variables_dynamic)
        self.var_n_static = len(self.variables_static)
        self.var_n = self.var_n_dynamic + self.var_n_static

    def __get_datacube(self):
        """
        Private method to get the dataset inside the root directory
        """

        with xr.open_dataset(os.path.join(self.root_datacube, "datacube_" + self.exp_name + ".nc")) as dataset:

            # care with slicing the cube if flipped along y-axis
            dataset = dataset.sel(time=slice(self.times[0], self.times[1])).isel(x=slice(self.x_min, self.x_max),
                                                                                 y=slice(self.y_min, self.y_max))

            self._datacube_dynamic = dataset[self.variables_dynamic].to_array().values.astype(np.float32)
            self._anomaly = dataset['anomaly_extreme'].sel(var=self.variables_dynamic).values
            self._extreme = dataset['extreme'].values

            self._datacube_static = None

            for v_s in self.variables_static:

                data = np.flip(dataset[v_s].values, -2)
                data = np.expand_dims(data, axis=0)

                data = (data - np.nanmean(data)) / np.nanstd(data)
                data = np.clip(data, a_min=-10., a_max=10.)

                if self._datacube_static is None:
                    self._datacube_static = data
                else:
                    self._datacube_static = np.vstack((self._datacube_static, data))

            self._datacube_timestep = np.arange(self.times[0], self.times[1] + 1, dtype=np.float32)  # absolute timestep
            self._datacube_time = np.zeros(self._datacube_dynamic.shape[1],
                                           dtype=np.float32)  # week number for each timestep

            for j, t in enumerate(self._datacube_timestep):
                self._datacube_time[j] = (t - 1) - 52 * ((t - 1) // 52) if (t - 1) // 52 != 0 else (t - 1)

            if self.is_replace_anomaly:
                # repalce abnormal events at extreme events with the normal distribution for each pixel
                #  self._datacube_dynamic[:, self._extreme > 0] = self.__median_pix_var[:, self._datacube_time.astype(np.int32), ...][:, self._extreme > 0]
                self._datacube_dynamic[:, self._extreme > 0] = np.random.normal(
                    self.__median_pix_var[:, self._datacube_time.astype(np.int32), ...][:, self._extreme > 0],
                    self.__std_pix_var[:, self._datacube_time.astype(np.int32), ...][:, self._extreme > 0])
        #           self._datacube_dynamic[self._anomaly_random > 0] = np.random.normal(
        #               self.__median_pix_var[:, self._datacube_time.astype(np.int32), ...][self._anomaly_random > 0],
        #               self.__std_pix_var[:, self._datacube_time.astype(np.int32), ...][self._anomaly_random > 0])
        #           self._anomaly_random = dataset['anomaly'].sel(var=self.variables_dynamic).values

        #           self._datacube_dynamic[self._anomaly > 0] = np.random.normal(
        #               self.__median_pix_var[:, self._datacube_time.astype(np.int32), ...][self._anomaly > 0],
        #               self.__std_pix_var[:, self._datacube_time.astype(np.int32), ...][self._anomaly > 0])

            # normalize artificial data
            if self.is_norm:
                if self.is_clima_scale:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        self._datacube_dynamic = (self._datacube_dynamic -
                                                  self.__median_pix_var[:, self._datacube_time.astype(np.int32), ...]) \
                                                 / self.__std_pix_var[:, self._datacube_time.astype(np.int32), ...]

                else:
                    for v in range(self.var_n_dynamic):
                        self._datacube_dynamic[v, ...] = (self._datacube_dynamic[v, ...] - self.__median_var[v]) \
                                                         / self.__std_var[v]

                self._datacube_dynamic = np.clip(self._datacube_dynamic, a_min=-10., a_max=10.)

            if self.is_reduce:
                self._datacube_static = np.nanmean(self._datacube_static.reshape((self.var_n_static,
                                                                                  self.n_lat_window,
                                                                                  self.window_size,
                                                                                  self.n_lon_window,
                                                                                  self.window_size)), axis=(2, 4))

                V, T, _, _ = self._datacube_dynamic.shape

                self._datacube_dynamic = np.nanmean(self._datacube_dynamic.reshape((V, T,
                                                                                    self.n_lat_window,
                                                                                    self.window_size,
                                                                                    self.n_lon_window,
                                                                                    self.window_size)),
                                                    axis=(3, 5))

                # check this
                self._anomaly = np.nanmean(self._anomaly.reshape((V, T,
                                                                  self.n_lat_window, self.window_size,
                                                                  self.n_lon_window, self.window_size)),
                                           axis=(3, 5))
                # check this
                self._extreme = np.nanmean(self._extreme.reshape((T,
                                                                  self.n_lat_window, self.window_size,
                                                                  self.n_lon_window, self.window_size)),
                                           axis=(2, 4))

    def __load_artificial_statistic(self):
        """
        Private method to get the statistics of the synthetic dataset from the root directory
        """

        with open(os.path.join(self.root_datacube, 'statistic_' + self.exp_name + '.json'), 'r') as file:
            dict_tmp = json.load(file)
            self.__min_var, self.__max_var, self.__mean_var, self.__median_var, self.__std_var = [], [], [], [], []

            for v in self.variables_dynamic:
                self.__min_var.append(float(dict_tmp['min'][v]))
                self.__max_var.append(float(dict_tmp['max'][v]))
                self.__mean_var.append(float(dict_tmp['mean'][v]))
                self.__median_var.append(float(dict_tmp['median'][v]))
                self.__std_var.append(float(dict_tmp['std'][v]))

            self.__min_var = np.array(self.__min_var)
            self.__max_var = np.array(self.__max_var)
            self.__mean_var = np.array(self.__mean_var)
            self.__median_var = np.array(self.__median_var)
            self.__std_var = np.array(self.__std_var)

    def __load_artificial_climatology_pixels(self):
        """
        Private method to get the weekly pixel-wise statistics of the synthetic dataset from the root directory
        """
        with xr.open_dataset(os.path.join(self.root_datacube, "climatology_" + self.exp_name + ".nc")) as dataset:
            #  self.__min_pix_var = dataset[self.variables_dynamic].sel(climatology="min"). \
            #      isel(x=slice(self.x_min, self.x_max), y=slice(self.y_min, self.y_max))
            #  self.__max_pix_var = dataset[self.variables_dynamic].sel(climatology="max"). \
            #      isel(x=slice(self.x_min, self.x_max), y=slice(self.y_min, self.y_max))
            #  self.__mean_pix_var = dataset[self.variables_dynamic].sel(climatology="mean"). \
            #      isel(x=slice(self.x_min, self.x_max), y=slice(self.y_min, self.y_max))
            self.__median_pix_var = dataset[self.variables_dynamic].sel(climatology="median"). \
                isel(x=slice(self.x_min, self.x_max), y=slice(self.y_min, self.y_max))
            self.__std_pix_var = dataset[self.variables_dynamic].sel(climatology="std"). \
                isel(x=slice(self.x_min, self.x_max), y=slice(self.y_min, self.y_max))

            self.__median_pix_var = self.__median_pix_var.to_array().values.astype(np.float32)
            self.__std_pix_var = self.__std_pix_var.to_array().values.astype(np.float32)


    def min_max_scale(self, array: np.array,
                      min_alt: float, max_alt: float,
                      min_new: float = 0., max_new: float = 1.):

        """
        Helper method to normalize an array between new minimum and maximum values

        Args:
        ----------
        array (np.array): array to be normalized
        min_alt (float): minimum value in array
        max_alt (float): maximum value in array
        min_new (float, optional): minimum value after normalization. Defaults to 0.
        max_new (float, optional): maximum value after normalization. Defaults to 1.

        Returns
        ----------
        array (np.array): normalized numpy array
        """

        array = ((max_new - min_new) * (array - min_alt) / (max_alt - min_alt)) + min_new

        return array

    def __getitem__(self, index):
        """
        Method to load datacube by the file index

        Args:
        ----------
        index (int): the index of the file

        Returns
        ----------
        datacube_dynamic (np.array): dynamic data [var_n_dynamic, channels, delta_t, n_lat_cut, n_lon]
        datacube_static (np.array): static data [var_n_static, channels, delta_t, n_lat_cut, n_lon]
        datacube_t (np.array): week number of the data [delta_t]
        mask_extreme (np.array): mask of extreme events [n_lat, n_lon]
        mask_extreme_loss (np.array): mask of extreme events for all time steps [n_lat, n_lon]
        mask_anomaly (np.array): mask of the anomalous events [var_n_dynamic, delta_t, n_lat, n_lon]
        mask_no_vegetation (np.array): mask of pixels without vegetation cover [n_lat, n_lon]
        datacube_tstep (np.array): time step of the file within the dataset
        """

        # load artificial data
        datacube_dynamic = np.flip(self._datacube_dynamic[:, index:index + self.delta_t, ...].copy(), 1)
        datacube_dynamic = np.expand_dims(datacube_dynamic, axis=1)
        # get static variables
        datacube_static = self._datacube_static.copy()
        # get data time
        datacube_t = np.flip(self._datacube_time[index:index + self.delta_t].copy() + 1)
        # get data timestep
        datacube_tstep = np.array([self._datacube_timestep[index + self.delta_t - 1].copy()])
        # load extreme mask
        # get masks for the extreme loss from the last time step 0
        # mask_extreme = np.flip(self._extreme.copy()[index: index + self.delta_t, ...], 0)
        # mask_extreme = self._extreme[index + self.delta_t - 1, ...].copy()
        # mask_extreme[mask_extreme > 1] = 0

        # get masks for the anomaly loss from all time steps
        mask_extreme_loss = np.flip(self._extreme[index: index + self.delta_t, ...].copy(), 0)
        #mask_extreme_loss = np.sum(mask_extreme_loss, axis=0)
        #mask_extreme_loss[mask_extreme_loss > 1] = 1
        #mask_extreme_loss = mask_extreme_loss.astype(np.uint8)

        # load anomaly mask
        mask_anomaly = np.flip(self._anomaly[:, index:index + self.delta_t, ...].copy(), 1)

        # augmentation
        if self.is_aug:
            is_rotate = np.random.choice(2)
            if is_rotate:
                # k = np.random.randint(1, 4)
                k = 2  # because of the rectangular dimensions
                datacube_dynamic = np.rot90(datacube_dynamic, k=k, axes=(-1, -2))
                datacube_static = np.rot90(datacube_static, k=k, axes=(-1, -2))
                #mask_extreme = np.rot90(mask_extreme, k=k, axes=(-1, -2))
                mask_extreme_loss = np.rot90(mask_extreme_loss, k=k, axes=(-1, -2))
                mask_anomaly = np.rot90(mask_anomaly, k=k, axes=(-1, -2))

            is_flip = np.random.choice(2)
            if is_flip:
                ax = np.random.randint(1, 3)
                datacube_dynamic = np.flip(datacube_dynamic, axis=-ax)
                datacube_static = np.flip(datacube_static, axis=-ax)
                #mask_extreme = np.flip(mask_extreme, axis=-ax)
                mask_extreme_loss = np.flip(mask_extreme_loss, axis=-ax)
                mask_anomaly = np.flip(mask_anomaly, axis=-ax)

        return datacube_dynamic.copy(), datacube_static.copy(), datacube_t.copy(), \
               mask_extreme_loss.copy(), mask_anomaly.copy(), datacube_tstep.copy()

    def __len__(self):
        """
        Method to get the number of files in the dataset
        Returns:
        ----------
        (int): the number of files in the dataset
        """
        return self._datacube_dynamic.shape[1] - self.delta_t + 1

    @property
    def datacube_dynamic(self):
        """ get the datacube dynamic data """
        return self._datacube_dynamic

    @property
    def extreme(self):
        """ get the datacube extreme events data """
        return self._extreme

    @property
    def anomaly(self):
        """ get the datacube anomalous events data """
        return self._anomaly

    @property
    def timestep(self):
        """ get the datacube time steps of the data """
        return self._datacube_timestep


if __name__ == '__main__':

    root_datacube = r'../Synthetic/synthetic_CERRA'

    variables_static = ['latitude', 'longitude']
    variables = ['var_01', 'var_03']
    variables.sort()
    times_val = (52 * 34 + 1, 52 * 40)

    dataset = Synthetic_Dataset(
        root_datacube=root_datacube, delta_t=8,
        is_aug=False, is_clima_scale=True, is_norm=True,
        variables=variables, variables_static=variables_static,
        times=times_val, window_size=1, is_replace_anomaly=False
    )

    print('number of sampled data:', dataset.__len__())

    is_test_run = False
    is_test_plot = True

    if is_test_run:

        import time
        import random

        manual_seed = 0
        random.seed(manual_seed)

        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=1,
                                                   # persistent_workers=False,
                                                   prefetch_factor=1)

        end = time.time()

        for i, (data_d, data_s, data_t, mask_extreme_loss, mask_anomaly, d_ts) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

    if is_test_plot:

        import matplotlib
        matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt

        n = [u'Δt$_{0}$', u'-Δt$_{1}$', u'-Δt$_{2}$', u'-Δt$_{3}$', u'-Δt$_{4}$', u'-Δt$_{5}$',
             u'-Δt$_{6}$', u'-Δt$_{7}$']

        for i in range(len(dataset)):

            #i = np.random.choice(len(dataset), 1, replace=False)
            data_d, data_s, data_t, data_extreme_loss, data_anomaly, data_ts = dataset[int(i)]

            for v in range(len(data_d)):

                fig, axs = plt.subplots(2, dataset.delta_t, figsize=(12, 4))
                for t in range(dataset.delta_t):

                    axs[0, t].imshow(data_d[v, 0, t, ...], cmap='cividis')
                    axs[0, t].set_title(n[t])
                    # axs[0, t].colorbar()

                    axs[1, t].imshow(data_anomaly[v, t, ...], cmap='cividis')
                    axs[1, t].set_title('anomaly ' + n[t])

                for ax in axs.flatten():
                    ax.set_axis_off()

                fig.suptitle(variables[v] + ', timestep=' +
                             str(int(data_ts[0])) + ', week=' + str(int(data_t[0])), y=0.96)
                plt.show()

