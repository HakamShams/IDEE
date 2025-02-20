# ------------------------------------------------------------------
"""
Dataset class for ERA5-Land ReAnalysis dataset

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
from datetime import datetime

np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
import dask

dask.config.set(scheduler='synchronous')

# ------------------------------------------------------------------


class ERA5_Land_Dataset(Dataset):
    """
        Dataset class for ERA5-Land ReAnalysis dataset

        Attributes
        ----------
        root_ERA5_Land (str): directory to ERA5-Land dataset
        root_NOAA (str): directory to NOAA dataset
        nan_fill (float): value to replace missing values
        delta_t (int): temporal resolution of the data
        is_aug (bool): option to use data augmentation
        is_shuffle (bool): option to shuffle the data
        is_clima_scale (bool): option to normalize the data with climatology
        is_norm (bool): option to normalize the data
        variables (list): list of ERA5-Land dynamic variable names
        variables_static (list, optional): list of ERA5-Land static variable names
        window_size (int): window to reduce resolution
        years (list): list of years
        threshold (int): threshold to define extreme drought events
                         values of Vegetation Health Index (VHI) below threshold will be considered extremes.
        alpha (float): alpha parameter to weight VCI and TCI. Defaults to 0.5
        region (str): name of the CORDEX region
        x_min (int): minimum longitude
        x_max (int): maximum longitude
        y_min (int): minimum latitude
        y_max (int): maximum latitude
        CORDEX_regions (dict): dimensions of the CORDEX regions
        is_reduce (bool): if the resolution is reduced
        n_lat (int): number of pixels in the latitude/y direction
        n_lon (int): number of pixels in the longitude/x direction
        n_lat_window (int): number of pixels in the latitude/y direction after reducing the resolution
        n_lon_window (int): number of pixels in the longitude/x direction after reducing the resolution
        var_n_dynamic (int): number of dynamic variables
        var_n (int): number of dynamic variables

        Methods
        -------
        __get_var_n()
            private method to get the number of variables
        __getpath()
            private method to get the files of the dataset inside the root directory

        __load_era5_land_statistic()
            private method to get the statistics of the Era5-Land dataset from the root directory
        __load_era5_land_climatology_pixels()
            private method to load ERA5-Land climatology from the root directory
        __load_valid_pixels_mask()
            private method to load the land/sea, non vegetation and prudence regions masks from the root directory
        __load_static_variables()
            private method to load the static variables from the root directory

        __generate_mask(NetCDF_file)
            private method to load NOAA data from the file path
        __load_datacube(NetCDF_file)
            private method to load ERA5-Land data from the file path
        _preprocess(x):
            helper function for the function __load_datacube()

        min_max_scale(array, min_alt, max_alt, min_new, max_new)
            helper method to normalize an array between new minimum and maximum values
        get_datacube_time(file)
            helper method to get the year, month, day and week number from the file name

        __getitem__(index)
            method to load datacube by the file index
        __len__()
            method to get the number of files
    """

    def __init__(self, root_ERA5_Land: str, root_NOAA: str, nan_fill: float = 0., delta_t: int = 4,
                 is_aug: bool = False, is_shuffle: bool = False, is_clima_scale: bool = False,
                 variables: list = None, window_size: int = 1, is_norm: bool = True,
                 years: list = None, threshold: float = 26.0,
                 alpha: float = 0.5, region: str = 'EUR-11',
                 x_min: int = None, x_max: int = None,
                 y_min: int = None, y_max: int = None
                 ):

        """
            Args:
            ----------
            root_ERA5_Land (str): directory to ERA5-Land dataset
            root_NOAA (str): directory to NOAA dataset
            nan_fill (float): value to replace missing values. Default to 0.
            delta_t (int, optional): temporal resolution of the data. Default to 4
            is_aug (bool, optional): option to use data augmentation. Defaults to False
            is_shuffle (bool, optional): option to shuffle the data. Defaults to False
            is_clima_scale (bool, optional): option to normalize the data with climatology. Defaults to False
            is_norm (bool, optional): option to normalize the data. Defaults to True
            variables (list, optional): list of ERA5-Land dynamic variable names. Defaults to None
            variables_static (list, optional): list of ERA5-Land static variable names. Defaults to None
            window_size (int, optional): window to reduce resolution. Defaults to 1
            years (list, optional): list of years. Defaults to None
            threshold (int, optional): threshold to define extreme drought events. Defaults to 26.
                                       values of Vegetation Health Index (VHI) below threshold will be considered extremes.
            alpha (float, optional): alpha parameter to weight VCI and TCI. Defaults to 0.5
            region (str, optional): name of the CORDEX region. Defaults to 'EUR-11'
            x_min (int, optional): minimum longitude. Defaults to 0
            x_max (int, optional): maximum longitude. Defaults to max(region)
            y_min (int, optional): minimum latitude. Defaults to 0
            y_max (int, optional): maximum latitude. Defaults to max(region)
        """

        super().__init__()

        self.region = region
        self.root_ERA5_Land = os.path.join(root_ERA5_Land, region)
        self.root_NOAA = os.path.join(root_NOAA, region)
        self.nan_fill = nan_fill
        self.delta_t = delta_t
        self.is_aug = is_aug
        self.is_shuffle = is_shuffle
        self.is_clima_scale = is_clima_scale
        self.is_norm = is_norm
        self.variables_dynamic = variables if variables is not None else ['d2m', 't2m', 'fal', 'sp', 'e',
                                                                          'tp', 'skt', 'stl1', 'swvl1']

        self.years = years if years is not None else [str(year) for year in range(1981, 2024+1)]

        self.threshold = threshold
        self.alpha = alpha

        self.window_size = window_size
        self.is_reduce = (window_size > 1)

        self.CORDEX_regions = {'EUR-11': (412, 424),
                               'AFR-11': (804, 776),
                               'NAM-11': (520, 620),
                               'SAM-11': (668, 584),
                               'CAS-11': (400, 612),
                               'EAS-11': (668, 812)
                               }
        if x_min is None:
            self.x_min = 0
        else:
            self.x_min = x_min
        if x_max is None:
            self.x_max = self.CORDEX_regions[region][1]
        else:
            self.x_max = x_max
        if y_min is None:
            self.y_min = 0
        else:
            self.y_min = y_min
        if y_max is None:
            self.y_max = self.CORDEX_regions[region][0]
        else:
            self.y_max = y_max

        self.n_lat = self.y_max - self.y_min
        self.n_lon = self.x_max - self.x_min

        self.n_lat_window = self.n_lat // window_size
        self.n_lon_window = self.n_lon // window_size

        # TODO add check for input

        # sort variables and years
        self.variables_dynamic.sort()
        self.years.sort()

        # preprocessing for the dataset
        self.__get_path()
        self.__get_var_n()

        if is_norm:
            if is_clima_scale:
                self.__load_era5_land_climatology_pixels()
            else:
                self.__load_era5_land_statistic()

        self.__load_valid_pixels_mask()

        if is_shuffle:
            np.random.shuffle(self.files)

    def __get_var_n(self):
        """
        Private method to get the number of variables
        """
        self.var_n_dynamic = len(self.variables_dynamic)
        self.var_n = self.var_n_dynamic

    def __get_path(self):
        """
        Private method to get the dataset files inside the root directory
        """

        self.files = []

        for year in self.years:

            year_dir_era5_land = os.path.join(self.root_ERA5_Land, year)
            if not os.path.isdir(year_dir_era5_land):
                raise ValueError('Year {} does not exist in the Era5-Land data'.format(year))
            year_dir_noaa = os.path.join(self.root_NOAA, year)
            if not os.path.isdir(year_dir_noaa):
                raise ValueError('Year {} does not exist in the NOAA data'.format(year))

            files = os.listdir(year_dir_noaa)
            files.sort()

            files = [file for file in files if file.endswith('.nc')]

            for week in range(52):

                week = week + 1
                week_nr = '0' + str(week) if week > 9 else '00' + str(week)

                files_noaa = [os.path.join(year_dir_noaa, file_noaa) for file_noaa in files if
                              file_noaa[-9:-6] == week_nr]

                if not files_noaa:
                    continue

                if int(year) == 1981 and week < (self.delta_t + 35):
                    continue
                if int(year) == 2024 and week > 14:
                    continue

                files_era5_land = []
                files_noaa = []
                weeks = []

                for week_t in range(self.delta_t):
                    week_t = week - week_t

                    if week_t > 0:
                        week_t_nr = '0' + str(week_t) if week_t > 9 else '00' + str(week_t)
                        file_era5_land = os.path.join(year_dir_era5_land, year + week_t_nr + '.nc')

                        file_noaa = [os.path.join(year_dir_noaa, file_noaa) for file_noaa in files if
                                     file_noaa[-9:-6] == week_t_nr]

                        if not file_noaa:
                            file_noaa = files_noaa[-1]

                    else:
                        week_t = week_t + 52
                        week_t_nr = '0' + str(week_t) if week_t > 9 else '00' + str(week_t)
                        year_t = str(int(year) - 1)

                        file_era5_land = os.path.join(os.path.join(self.root_ERA5_Land, year_t), year_t + week_t_nr + '.nc')

                        year_dir_noaa_t = os.path.join(self.root_NOAA, year_t)
                        files_t = os.listdir(year_dir_noaa_t)
                        files_t.sort()
                        files_t = [file for file in files_t if file.endswith('.nc')]

                        file_noaa = [os.path.join(year_dir_noaa_t, file_noaa) for file_noaa in files_t if
                                     file_noaa[-9:-6] == week_t_nr]

                        if not file_noaa:
                            file_noaa = files_noaa[-1]

                    files_era5_land.append(file_era5_land)
                    files_noaa.append(file_noaa)
                    weeks.append(week_t)

                self.files.append((files_era5_land, files_noaa, np.array(weeks).astype(np.float32)))  # all dataset

        if len(self.files) == 0:
            raise ValueError('No files were found in the root directories')

    def __load_era5_land_statistic(self):
        """
        Private method to get the statistics of the ERA5-Land dataset from the root directory
        """

        with open(os.path.join(self.root_ERA5_Land, self.region + '_statistic_train.json'), 'r') as file:
            dict_tmp = json.load(file)

            self.__min_var, self.__max_var, self.__mean_var, self.__std_var = [], [], [], []

            for v in self.variables_dynamic:
                self.__min_var.append(float(dict_tmp['min'][v]))
                self.__max_var.append(float(dict_tmp['max'][v]))
                self.__mean_var.append(float(dict_tmp['mean'][v]))
                self.__std_var.append(float(dict_tmp['std'][v]))

            self.__min_var = np.array(self.__min_var)
            self.__max_var = np.array(self.__max_var)
            self.__mean_var = np.array(self.__mean_var)
            self.__std_var = np.array(self.__std_var)

    def __load_era5_land_climatology_pixels(self):
        """
        Private method to get the weekly pixel-wise statistics of the ERA5-Land dataset from the root directory
        """
        with xr.open_dataset(os.path.join(self.root_ERA5_Land, self.region + "_climatology_pixels_train.nc")) as dataset_climatology:
               #self.__min_pix_var = dataset_climatology[self.variables_dynamic].sel(climatology="min"). \
               #    isel(rlon=slice(self.x_min, self.x_max), rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max, self.CORDEX_regions[self.region][0] - self.y_min))
               #self.__max_pix_var = dataset_climatology[self.variables_dynamic].sel(climatology="max"). \
               #    isel(rlon=slice(self.x_min, self.x_max), rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max, self.CORDEX_regions[self.region][0] - self.y_min))
               self.__mean_pix_var = dataset_climatology[self.variables_dynamic].sel(climatology="mean").\
                   isel(rlon=slice(self.x_min, self.x_max), rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max, self.CORDEX_regions[self.region][0] - self.y_min))
               #self.__median_pix_var = dataset_climatology[self.variables_dynamic].sel(climatology="median").\
               #    isel(rlon=slice(self.x_min, self.x_max), rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max, self.CORDEX_regions[self.region][0] - self.y_min))
               self.__std_pix_var = dataset_climatology[self.variables_dynamic].sel(climatology="std").\
                   isel(rlon=slice(self.x_min, self.x_max), rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max, self.CORDEX_regions[self.region][0] - self.y_min))

    def __load_valid_pixels_mask(self):
        """
        Private method to load the land/sea and non vegetation masks from the root directory
        """

        with xr.open_dataset(os.path.join(self.root_NOAA, self.region + "_masks.nc")) as dataset_valid_pixels:
            self.__mask_no_vegetation = dataset_valid_pixels['mask_no_vegetation']. \
                isel(rlon=slice(self.x_min, self.x_max), rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max,
                                                                    self.CORDEX_regions[self.region][0] - self.y_min)).values
            self.__mask_no_vegetation = np.flip(self.__mask_no_vegetation, axis=-2).astype(np.float32)

            self.__mask_water = dataset_valid_pixels['lsm'].isel(rlon=slice(self.x_min, self.x_max),
                                                                 rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max,
                                                                            self.CORDEX_regions[self.region][0] - self.y_min)).values
            self.__mask_water = np.flip(self.__mask_water, axis=-2).astype(np.float32)
            self.__mask_water = -1 * (self.__mask_water - 1)  # water is 1 and land is 0

            if self.is_reduce:
                self.__mask_no_vegetation_scaled = np.nanmean(self.__mask_no_vegetation.reshape((self.n_lat_window,
                                                                                                 self.window_size,
                                                                                                 self.n_lon_window,
                                                                                                 self.window_size)),
                                                              axis=(1, 3))
                self.__mask_no_vegetation_scaled[self.__mask_no_vegetation_scaled >= 0.5] = 1
                self.__mask_no_vegetation_scaled[self.__mask_no_vegetation_scaled < 0.5] = 0

                self.__mask_water_scaled = np.nanmean(self.__mask_water.reshape((self.n_lat_window,
                                                                                 self.window_size,
                                                                                 self.n_lon_window,
                                                                                 self.window_size)),
                                                      axis=(1, 3))

                self.__mask_water_scaled[self.__mask_water_scaled >= 0.5] = 1
                self.__mask_water_scaled[self.__mask_water_scaled < 0.5] = 0

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

    def get_datacube_time(self, file: str):
        """
        Helper method to get the year, month, day, week number, and day of the year from the file name

        Args:
        ----------
        file (str): name of the file in the dataset

        Returns
        ----------
        year (int): corresponding year for the file
        month (int): corresponding month number for the file
        day (int): corresponding day of the month for the file
        week (int): corresponding week number of the file
        day_of_year (int): corresponding day of the year of the file
        """

        file_name = os.path.splitext(os.path.basename(os.path.normpath(file)))[0]

        year = int(file_name[:4])
        month = int(file_name[4:6])
        day = int(file_name[6:])

        day_of_year = datetime(year, month, day).timetuple().tm_yday
        week = np.min([(day_of_year - 1) // 7 + 1, 52])

        return year, month, day, week, day_of_year

    def __generate_mask(self, NetCDF_files: list | str, thr: float = 26):
        """
        Private method to load NOAA data from the file path

        Args:
        ----------
        NetCDF_file (list or str): the file path/es
        thr (float, optional): threshold to define extreme drought events. Defaults to 26.
                               values of Vegetation Health Index (VHI) below threshold will be masked to 1

        Returns
        ----------
        mask_drought (np.array): extreme agricultural drought events where VHI < threshold [n_lat, n_lon]
        cold_surface (np.array): cold surfaces [n_lat, n_lon]
        """

        VHI, cold_surface = None, None

        for NetCDF_file in NetCDF_files:

            VHI_s = self.alpha * xr.load_dataset(NetCDF_file)['VCI'].isel(rlon=slice(self.x_min, self.x_max),
                                                                          rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max,
                                                                                  self.CORDEX_regions[self.region][0] - self.y_min)).values \
                    + (1 - self.alpha) * xr.load_dataset(NetCDF_file)['TCI'].isel(rlon=slice(self.x_min, self.x_max),
                                                                                  rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max,
                                                                                          self.CORDEX_regions[self.region][0] - self.y_min)).values

            cold_surface_s = xr.load_dataset(NetCDF_file)['mask_cold_surface'].isel(rlon=slice(self.x_min, self.x_max),
                                                                                    rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max,
                                                                                            self.CORDEX_regions[self.region][0] - self.y_min)).values

            if VHI is None:
                VHI = np.expand_dims(VHI_s, 0)
                cold_surface = np.expand_dims(cold_surface_s, 0)
            else:
                VHI = np.concatenate((VHI, VHI_s[None, :, :]), axis=0)
                cold_surface = np.concatenate((cold_surface, cold_surface_s[None, :, :]), axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            VHI = np.nanmean(VHI, axis=0, keepdims=False)

            cold_surface = np.flip(cold_surface.sum(axis=0), 0).astype(np.float32)
            cold_surface[cold_surface > 1] = 1

            if self.is_reduce:
                VHI = np.nanmean(VHI.reshape((self.n_lat_window, self.window_size,
                                              self.n_lon_window, self.window_size)), axis=(1, 3))

        mask_drought = (np.flip(VHI, 0) < thr).astype(np.float32)

        if self.is_reduce:
            cold_surface = np.nanmin(cold_surface.reshape((self.n_lat_window, self.window_size,
                                                           self.n_lon_window, self.window_size)), axis=(1, 3))
            cold_surface[cold_surface >= 0.5] = 1
            cold_surface[cold_surface < 0.5] = 0

        mask_drought[np.isnan(mask_drought)] = 0
        mask_drought[cold_surface == 1] = 0

        if self.is_reduce:
            mask_drought[self.__mask_no_vegetation_scaled == 1] = 0
            mask_drought[self.__mask_water_scaled == 1] = 0
        else:
            mask_drought[self.__mask_no_vegetation == 1] = 0
            mask_drought[self.__mask_water == 1] = 0

        return mask_drought, cold_surface

    def _preprocess(self, x):
        """ Helper function for the function __load_datacube() """
        return x.isel(rlon=slice(self.x_min, self.x_max),
                      rlat=slice(self.CORDEX_regions[self.region][0] - self.y_max,
                              self.CORDEX_regions[self.region][0] - self.y_min))[
            self.variables_dynamic].reset_coords(drop=True)

    def __load_datacube(self, NetCDF_files):
        """
        Private method to load ERA5-Land data from the file path

        Aegs:
        ----------
        NetCDF_file (str): the file path/es

        Returns
        ----------
        datacube (np.array): ERA5-Land dynamic data [var_n_dynamic, channels, delta_t, n_lat, n_lon]
        """

        datacube = xr.open_mfdataset(NetCDF_files,
                                     # combine='by_coords',
                                     combine='nested',
                                     concat_dim='None',
                                     preprocess=self._preprocess,
                                     parallel=True,
                                     #            decode_times=False,
                                     #           decode_cf=False,
                                     engine='netcdf4',
                                     ).sel(statistic=['mean', 'std']).to_array().values

        datacube = np.swapaxes(datacube, 1, 2)

        return np.flip(datacube, -2).astype(np.float32)

    def __getitem__(self, index):
        """
        Method to load datacube by the file index

        Args:
        ----------
        index (int): the index of the file

        Returns
        ----------
        datacube_dynamic (np.array): ERA5-Land dynamic data [var_n_dynamic, channels, delta_t, n_lat_cut, n_lon]
        mask_drought (np.array): mask of extreme agricultural droughts [n_lat, n_lon]
        mask_drought_loss (np.array): mask of extreme agricultural droughts for all time steps [n_lat, n_lon]
        mask_cold_surface (np.array): mask of very cold surface and invalid pixels [n_lat, n_lon]
        mask_cold_surface_loss (np.array): mask of very cold surface and invalid pixels for all time steps [n_lat, n_lon]
        mask_sea (np.array): sea/water mask [n_lat, n_lon]
        mask_no_vegetation (np.array): mask of pixels without vegetation cover [n_lat, n_lon]
        files_era5_land (str): name of the file
        """

        # get the ERA5-Land and NOAA files to be loaded as week number
        files_era5_land, files_noaa, datacube_t = self.files[index]

        # load ERA5-Land data
        # weeks have the same order as in files_era5_land >> target week has the order 0 in datacube_dynamic
        datacube_dynamic = self.__load_datacube(files_era5_land)

        # load NOAA data

        # get masks for the anomaly loss from all time steps
        mask_drought_loss = np.zeros((self.delta_t, self.n_lat_window, self.n_lon_window), dtype=np.float32)
        mask_cold_surface_loss = np.zeros((self.delta_t, self.n_lat_window, self.n_lon_window), dtype=np.float32)

        for delta_index in range(self.delta_t):
            mask_drought_loss[delta_index], mask_cold_surface_loss[delta_index] = self.__generate_mask(files_noaa[delta_index],
                                                                                                       thr=35)

        mask_drought_loss = np.sum(mask_drought_loss, axis=0, keepdims=False)
        mask_drought_loss[mask_drought_loss > 1] = 1

        # get masks for the extreme loss from the last time step 0
        mask_drought, mask_cold_surface = self.__generate_mask(files_noaa[0], thr=self.threshold)

        # get land/sea and non_vegetation masks
        if self.is_reduce:
            mask_sea, mask_no_vegetation = self.__mask_water_scaled.copy(), self.__mask_no_vegetation_scaled.copy()
        else:
            mask_sea, mask_no_vegetation = self.__mask_water.copy(), self.__mask_no_vegetation.copy()

        # normalize ERA5-Land data
        if self.is_norm:
            if self.is_clima_scale:
                mean_pix_var = np.flip(self.__mean_pix_var.sel(week=datacube_t).to_array().values, -2)
                std_pix_var = np.flip(self.__std_pix_var.sel(week=datacube_t).to_array().values, -2)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    datacube_dynamic = (datacube_dynamic - mean_pix_var) / std_pix_var
            else:
                for v in range(self.var_n_dynamic):
                    datacube_dynamic[v, 0, ...] = (datacube_dynamic[v, 0, ...] - self.__mean_var[v]) / self.__std_var[v]
                    datacube_dynamic[v, 1, ...] = (datacube_dynamic[v, 1, ...]) / self.__std_var[v]

            datacube_dynamic = np.clip(datacube_dynamic, a_min=-10., a_max=10.)

        # fill in the missing data
        datacube_dynamic[np.logical_or(np.isnan(datacube_dynamic), np.isinf(datacube_dynamic))] = self.nan_fill

        # reduce dimensions
        if self.is_reduce:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                datacube_dynamic = np.nanmean(datacube_dynamic.reshape((self.var_n_dynamic, 2, self.delta_t,
                                                                        self.n_lat_window, self.window_size,
                                                                        self.n_lon_window, self.window_size)), axis=(4, 6))

        # augmentation
        if self.is_aug:
            is_rotate = np.random.choice(2)
            if is_rotate:
                # k = np.random.randint(1, 4)
                k = 2
                datacube_dynamic = np.rot90(datacube_dynamic, k=k, axes=(-1, -2))
                mask_drought = np.rot90(mask_drought, k=k, axes=(-1, -2))
                mask_drought_loss = np.rot90(mask_drought_loss, k=k, axes=(-1, -2))
                mask_cold_surface = np.rot90(mask_cold_surface, k=k, axes=(-1, -2))
                mask_cold_surface_loss = np.rot90(mask_cold_surface_loss, k=k, axes=(-1, -2))
                mask_sea = np.rot90(mask_sea, k=k, axes=(-1, -2))
                mask_no_vegetation = np.rot90(mask_no_vegetation, k=k, axes=(-1, -2))

            is_flip = np.random.choice(2)
            if is_flip:
                ax = np.random.randint(1, 3)
                datacube_dynamic = np.flip(datacube_dynamic, axis=-ax)
                mask_drought = np.flip(mask_drought, axis=-ax)
                mask_drought_loss = np.flip(mask_drought_loss, axis=-ax)
                mask_cold_surface = np.flip(mask_cold_surface, axis=-ax)
                mask_cold_surface_loss = np.flip(mask_cold_surface_loss, axis=-ax)
                mask_sea = np.flip(mask_sea, axis=-ax)
                mask_no_vegetation = np.flip(mask_no_vegetation, axis=-ax)

        return datacube_dynamic.copy(), mask_drought.copy(), mask_drought_loss.copy(), \
               mask_cold_surface.copy(), mask_cold_surface_loss.copy(), \
               mask_sea.copy(), mask_no_vegetation.copy(), \
               files_era5_land[0][-10:-3]


    def __len__(self):
        """
        Method to get the number of files in the dataset
        Returns:
        ----------
        (int): the number of files in the dataset
        """
        return len(self.files)


if __name__ == '__main__':

    root_ERA5_land = r'../ERA5-Land'
    root_NOAA = r'../NOAA_CORDEX'

    variables = ['t2m', 'fal', 'e', 'tp', 'stl1', 'swvl1']

    variables.sort()

    years_train = [str(year) for year in range(1981, 2018)]

    dataset = ERA5_Land_Dataset(
        root_ERA5_Land=root_ERA5_land, root_NOAA=root_NOAA, nan_fill=0., is_aug=False, is_shuffle=False,
        is_clima_scale=False, variables=variables, years=years_train, is_norm=False,
        threshold=26.0, window_size=1, delta_t=4, alpha=0.5, region='EUR-11',
        #x_min=140, x_max=240, y_min=160, y_max=260
        x_min=None, x_max=None, y_min=None, y_max=None
    )

    print('number of sampled data:', dataset.__len__())

    is_test_run = True
    is_test_plot = False

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
                                                   num_workers=8,
                                                   #persistent_workers=False,
                                                   prefetch_factor=1)

        end = time.time()


        for i, (data_d, data_drought, data_drought_loss, data_cold_surface, data_cold_surface_loss,
                data_sea, data_no_vegetation, file_name) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

    if is_test_plot:

        import matplotlib
        matplotlib.use('TkAgg')

        var_t = ['mean', 'std']

        for i in range(len(dataset)):
            #i = np.random.choice(len(dataset), 1, replace=False)
            #i += 32
            data_d, data_drought, data_drought_loss, mask_cold_surface, mask_cold_surface_loss, data_sea, data_no_vegetation, file_name = dataset[int(i)]

            for v in range(data_d.shape[0]):
                for j in range(data_d.shape[1]):
                    #print(variables[v])
                    for t in range(dataset.delta_t):
                        #print(np.unique(data_d[v, j, t, :, :]))
                        plt.imshow(data_d[v, j, t, :, :], cmap='cividis')
                        plt.title(variables[v] + ' ' + var_t[j] + ' - delta_t: ' + str(t))
                        plt.colorbar()
                        plt.axis('off')
                        plt.show()
                        plt.close()