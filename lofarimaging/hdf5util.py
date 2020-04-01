""" Functions for working with single station data stored in HDF5

The HDF5 format used is the following:

obs000001          A group per observation (numbering is arbitrary)
  xst_data         Uncalibrated data as a full matrix of complex numbers
                   Ordering is by antenna number, like the XST files are dumped
                   (so typically first an x-pol, then a y-pol, then x-pol etc)
  calibrated_data  Calibrated data as a full matrix of complex numbers
                   Same ordering as xst_data
  sky_img          Sky image data as matrix of real numbers
  ground_imgs      Group for ground images
    ground_img000  Ground image as matrix of real numbers

Per observation, the following attributes are used:
 * frequency       Frequency in Hz
 * obstime         Observation time (UTC)
 * rcu_mode        RCU mode
 * station_name    Station name, e.g. CS103
 * subband         Subband

 For ground images, the following additional attributes are used:
 * extent          Extent of image, in metres from station center
                   [x_min, x_max, y_min, y_max]
 * extent_lonlat   Extent of image in longitude and latitude (w.r.t. WGS84 ellipsoid)
                   [lon_min, lon_max, lat_min, lat_max]
 * height          Height (w.r.t. station phase centre) of the image plane in metres
"""

import datetime
from typing import List
import numpy as np
import h5py

__all__ = ["get_new_obsname", "write_hdf5", "merge_hdf5"]


def get_new_obsname(h5file: h5py.File):
    """
    Get the next available observation name for a HDF5 file

    Args:
        h5file: HDF5 file with groups called' obs000001 etc

    Returns:
        str: "obs000002" etc

    Example:
        >>> emptyfile = h5py.File("test/empty.h5")
        >>> get_new_obsname(emptyfile)
        'obs000001'
    """
    all_obsnums = [int(obsname[3:]) for obsname in h5file]
    if len(all_obsnums) == 0:
        new_obsnum = 1
    else:
        new_obsnum = max(all_obsnums) + 1
    return f"obs{new_obsnum:06d}"


def write_hdf5(filename: str, xst_data: np.ndarray, visibilities: np.ndarray, sky_img: np.ndarray,
               ground_img: np.ndarray, station_name: str, subband: int, rcu_mode: int, frequency: float,
               obstime: datetime.datetime, extent: List[float], extent_lonlat: List[float],
               height: float):
    """
    Write an HDF5 file with all data

    Args:
        filename (str): Output filename. Will be appended to if a file already exists.
        xst_data (np.ndarray): Raw uncalibrated data (shape [n_ant, n_ant])
        visibilities (np.ndarray): Calibrated data (shape [n_ant, n_ant])
        sky_img (np.ndarray): Sky image as array
        ground_img (np.ndarray): Ground image as array
        station_name (str): Station name
        subband (int): Subband number
        rcu_mode (int): RCU mode
        frequency (float): Frequency
        obstime (datetime.datetime): Time of observation
        extent (List[float]): Extent of ground image in XY coordinates around station center
        extent_lonlat (List[float]): Extent of ground image in long lat coordinates
        height (float): Height of ground image (in metres)

    Returns:
        None

    Example:
        >>> xst_data = visibilities = np.ones((96, 96), dtype=np.complex)
        >>> ground_img = sky_img = np.ones((131, 131), dtype=np.float)
        >>> write_hdf5("test/test.h5", xst_data, visibilities, sky_img, ground_img, "DE603", \
                       297, 3, 150e6, datetime.datetime.now(), [-150, 150, -150, 150], \
                       [11.709, 11.713, 50.978, 50.981], 1.5)
    """
    short_station_name = station_name[:5]

    with h5py.File(filename, 'a') as h5file:
        new_obsname = get_new_obsname(h5file)
        obs_group = h5file.create_group(new_obsname)
        obs_group.attrs["obstime"] = str(obstime)[:19]
        obs_group.attrs["rcu_mode"] = rcu_mode
        obs_group.attrs["frequency"] = frequency
        obs_group.attrs["subband"] = subband
        obs_group.attrs["station_name"] = short_station_name

        obs_group.create_dataset("xst_data", data=xst_data, compression="gzip")
        obs_group.create_dataset("calibrated_data", data=visibilities, compression="gzip")
        obs_group.create_dataset("sky_img", data=sky_img, compression="gzip")

        ground_img_group = obs_group.create_group("ground_images")
        dataset_ground_img = ground_img_group.create_dataset("ground_img000", data=ground_img, compression="gzip")
        dataset_ground_img.attrs["extent"] = extent
        dataset_ground_img.attrs["extent_lonlat"] = extent_lonlat
        dataset_ground_img.attrs["height"] = height


def merge_hdf5(src_filename: str, dest_filename: str):
    """
    Merge HDF5 files containing groups with observations called obs000001 etc.
    Observations from src_filename will be appended to dest_filename, the obs
    numbers will be changed.

    Args:
        src_filename: Source filename
        dest_filename: Destination filename

    Returns:
        None

    Example:
        >>> import h5py
        >>> with h5py.File("test/test_src.h5", 'w') as src_file:
        ...     src_file.create_group("obs00001")
        ...     src_file.create_group("obs00002")
        <HDF5 group "/obs00001" (0 members)>
        <HDF5 group "/obs00002" (0 members)>
        >>> with h5py.File("test/test_dest.h5", 'w') as dest_file:
        ...     dest_file.create_group("obs000005")
        <HDF5 group "/obs000005" (0 members)>
        >>> merge_hdf5("test/test_src.h5", "test/test_dest.h5")
        >>> list(h5py.File("test/test_dest.h5"))
        ['obs000005', 'obs000006', 'obs000007']
    """
    with h5py.File(dest_filename) as dest_file:
        with h5py.File(src_filename, 'r') as src_file:
            for src_obsname in src_file:
                dest_obsname = get_new_obsname(dest_file)
                h5py.h5o.copy(src_file.id, bytes(src_obsname, 'utf-8'),
                              dest_file.id, bytes(dest_obsname, 'utf-8'))
                dest_file.flush()
