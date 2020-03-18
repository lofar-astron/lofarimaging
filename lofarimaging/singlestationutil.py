"""Functions for working with LOFAR single station data"""

import numpy as np
import os
import datetime
import lofargeotiff

from lofarantpos.db import LofarAntennaDatabase

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
import matplotlib.axes as maxes

from astropy.coordinates import SkyCoord, GCRS, EarthLocation, AltAz, get_sun
import astropy.units as u
from astropy.time import Time

from typing import List, Dict, Any, Tuple, Union

import lofarantpos
from packaging import version

from .maputil import get_map, make_leaflet_map
from .lofarimaging import nearfield_imager, sky_imager, skycoord_to_lmn


__all__ = ["sb_from_freq", "freq_from_sb", "find_caltable", "read_caltable",
           "rcus_in_station", "read_acm_cube", "get_station_pqr", "get_station_type",
           "make_sky_plot", "make_ground_plot", "make_xst_plots", "apply_calibration"]

__version__ = "1.5.0"

# Configurations for HBA observations with a single dipole activated per tile.
GENERIC_INT_201512 = [0, 5, 3, 1, 8, 3, 12, 15, 10, 13, 11, 5, 12, 12, 5, 2, 10, 8, 0, 3, 5, 1, 4, 0, 11, 6, 2, 4, 9,
                      14, 15, 3, 7, 5, 13, 15, 5, 6, 5, 12, 15, 7, 1, 1, 14, 9, 4, 9, 3, 9, 3, 13, 7, 14, 7, 14, 2, 8,
                      8, 0, 1, 4, 2, 2, 12, 15, 5, 7, 6, 10, 12, 3, 3, 12, 7, 4, 6, 0, 5, 9, 1, 10, 10, 11, 5, 11, 7, 9,
                      7, 6, 4, 4, 15, 4, 1, 15]
GENERIC_CORE_201512 = [0, 10, 4, 3, 14, 0, 5, 5, 3, 13, 10, 3, 12, 2, 7, 15, 6, 14, 7, 5, 7, 9, 0, 15, 0, 10, 4, 3, 14,
                       0, 5, 5, 3, 13, 10, 3, 12, 2, 7, 15, 6, 14, 7, 5, 7, 9, 0, 15]
GENERIC_REMOTE_201512 = [0, 13, 12, 4, 11, 11, 7, 8, 2, 7, 11, 2, 10, 2, 6, 3, 8, 3, 1, 7, 1, 15, 13, 1, 11, 1, 12, 7,
                         10, 15, 8, 2, 12, 13, 9, 13, 4, 5, 5, 12, 5, 5, 9, 11, 15, 12, 2, 15]

assert (version.parse(lofarantpos.__version__) >= version.parse("0.4.0"))


def sb_from_freq(freq: float, rcu_mode: Union[int, str] = 1) -> int:
    """
    Convert subband number to central frequency

    Args:
        rcu_mode: rcu mode
        freq: frequency in Hz

    Returns:
        int: subband number

    Example:
        >>> sb_from_freq(58007812.5, '3')
        297
    """
    clock = 200e6
    if int(rcu_mode) == 6:
        clock = 160e6

    freq_offset = 0
    if int(rcu_mode) == 5:
        freq_offset = 100e6
    elif int(rcu_mode) == 6:
        freq_offset = 160e6
    elif int(rcu_mode) == 7:
        freq_offset = 200e6

    sb_bandwidth = 0.5 * clock / 512.
    sb = round((freq - freq_offset) / sb_bandwidth)
    return int(sb)


def freq_from_sb(sb: int, rcu_mode: Union[str, int] = 1):
    """
    Convert central frequency to subband number

    Args:
        rcu_mode: rcu mode
        sb: subband number

    Returns:
        float: frequency in Hz

    Example:
        >>> freq_from_sb(297, '3')
        58007812.5
    """
    clock = 200e6
    if int(rcu_mode) == 6:
        clock = 160e6

    freq_offset = 0
    if int(rcu_mode) == 5:
        freq_offset = 100e6
    elif int(rcu_mode) == 6:
        freq_offset = 160e6
    elif int(rcu_mode) == 7:
        freq_offset = 200e6

    sb_bandwidth = 0.5 * clock / 512.
    freq = (sb * sb_bandwidth) + freq_offset
    return freq


def find_caltable(field_name: str, rcu_mode: Union[str, int], caltable_dir='caltables'):
    """
    Find the file of a caltable.

    Args:
        field_name: Name of the antenna field, e.g. 'DE602LBA' or 'DE602'
        rcu_mode: Receiver mode for which the calibration table is requested.
            Probably should be  'inner' or 'outer'
        caltable_dir: Root directory under which station information is stored in
            subdirectories DE602C/etc/, RS106/etc/, ...
    Returns:
        str: full path to caltable if it exists, None if nothing found

    Example:
        >>> find_caltable("DE603LBA", "3", caltable_dir="test/CalTables")
        'test/CalTables/DE603/CalTable-603-LBA_INNER-10_90.dat'

        >>> find_caltable("ES615HBA", "5") is None
        True
    """
    station, field = field_name[0:5].upper(), field_name[5:].upper()
    station_number = station[2:5]

    filename = f"CalTable-{station_number}"

    if str(rcu_mode) in ('outer', '1', '2'):
        filename += "-LBA_OUTER-10_90.dat"
    elif str(rcu_mode) in ('inner', '3', '4'):
        filename += "-LBA_INNER-10_90.dat"
    elif str(rcu_mode) == '5':
        filename += "-HBA-110_190.dat"
    elif str(rcu_mode) == '6':
        filename += "-HBA-170_230.dat"
    elif str(rcu_mode) == '7':
        filename += "-HBA-210_250.dat"
    else:
        raise RuntimeError("Unexpected mode: " + str(rcu_mode) + " for field_name " + str(field_name))

    if os.path.exists(os.path.join(caltable_dir, filename)):
        # All caltables in one directory
        return os.path.join(caltable_dir, filename)
    elif os.path.exists(os.path.join(caltable_dir, station, filename)):
        # Caltables in a directory per station
        return os.path.join(caltable_dir, station, filename)
    else:
        return None


def read_caltable(filename: str, num_subbands=512) -> Tuple[Dict[str, str], np.ndarray]:
    """
    Read a station's calibration table.

    Args:
        filename: Filename with the caltable
        num_subbands: Number of subbands

    Returns:
        Tuple[Dict[str, str], np.ndarray]: A tuple containing a dict with
            the header lines, and a 2D numpy.array of complex numbers
            representing the station gain coefficients.
    """
    infile = open(filename, 'rb')

    header_lines = []

    try:
        while True:
            header_lines.append(infile.readline().decode('utf8').strip())
            if 'HeaderStop' in header_lines[-1]:
                break
    except UnicodeDecodeError:
        # No header; close and open again
        infile.close()
        infile = open(filename, 'rb')

    caldata = np.fromfile(infile, dtype=np.complex128)
    num_rcus = len(caldata) // num_subbands

    infile.close()

    header_dict = {key: val for key, val in [line.split(" = ")
                                             for line in header_lines[1:-1]]}

    return header_dict, caldata.reshape((num_subbands, num_rcus))


def apply_calibration(visibilities: np.ndarray, station_name: str, rcu_mode: Union[str, int],
                      subband: int, caltable_dir: str = "CalTables"):
    """
    Apply calibration to visibilities

    Args:
        visibilities (np.ndarray): Visibility cube
        station_name (str): Station name, e.g. "DE603"
        rcu_mode (Union[str, int]): RCU mode, e.g. 5
        subband (int): Subband
        caltable_dir (str, optional): Directory with calibration tables. Defaults to "CalTables".

    Returns:
        Tuple[np.ndarray, Dict[str, str]]: modified visibilities and dictionary with calibration info
    """
    caltable_filename = find_caltable(station_name, rcu_mode=rcu_mode,
                                      caltable_dir=caltable_dir)
    cal_header = {}
    if caltable_filename is None:
        print('No calibration table found... cube remains uncalibrated!')
    else:
        cal_header, cal_data = read_caltable(caltable_filename)

        rcu_gains = cal_data[subband, :]
        rcu_gains = np.array(rcu_gains, dtype=np.complex64)
        gain_matrix = rcu_gains[np.newaxis, :] * np.conj(rcu_gains[:, np.newaxis])
        visibilities = visibilities / gain_matrix

    calibration_info = {}
    if "CalTableHeader.Observation.Date" in cal_header:
        calibration_info["calibration_obsdate"] = cal_header["CalTableHeader.Observation.Date"]
    if "CalTableHeader.Calibration.Date" in cal_header:
        calibration_info["calibration_date"] = cal_header["CalTableHeader.Calibration.Date"]
    if "CalTableHeader.Comment" in cal_header:
        calibration_info["calibration_comment"] = cal_header["CalTableHeader.Comment"]
    if caltable_filename is not None:
        calibration_info["calibration_filename"] = caltable_filename

    return visibilities, calibration_info

def rcus_in_station(station_type: str):
    """
    Give the number of RCUs in a station, given its type.

    Args:
        station_type: Kind of station that produced the correlation. One of
            'core', 'remote', 'intl'.

    Example:
        >>> rcus_in_station('remote')
        96
    """
    return {'core': 96, 'remote': 96, 'intl': 192}[station_type]


def read_acm_cube(filename: str, station_type: str):
    """
    Read an ACM binary data cube (function from Michiel)

    Args:
        filename: File containing the array correlation matrix.
        station_type: Kind of station that produced the correlation. One of
            'core', 'remote', 'intl'.

    Returns:
        np.array: 3D cube of complex numbers, with indices [time slots, rcu, rcu].

    Example:
        >>> cube = read_acm_cube('test/20170720_095816_mode_3_xst_sb297.dat', 'intl')
        >>> cube.shape
        (29, 192, 192)
    """
    num_rcu = rcus_in_station(station_type)
    data = np.fromfile(filename, dtype=np.complex128)
    time_slots = int(len(data) / num_rcu / num_rcu)
    return data.reshape((time_slots, num_rcu, num_rcu))


def get_station_type(station_name: str) -> str:
    """
    Get the station type, one of 'intl', 'core' or 'remote'

    Args:
        station_name: Station name, e.g. "DE603LBA" or just "DE603"

    Returns:
        str: station type, one of 'intl', 'core' or 'remote'

    Example:
        >>> get_station_type("DE603LBA")
        'intl'
    """
    if station_name[0] == "C":
        return "core"
    elif station_name[0] == "R" or station_name[:5] == "PL611":
        return "remote"
    else:
        return "intl"


def get_station_pqr(station_name: str, rcu_mode: Union[str, int], db):
    """
    Get PQR coordinates for the relevant subset of antennas in a station.

    Args:
        station_name: Station name, e.g. DE603LBA
        rcu_mode: RCU mode (0 - 6, can be string)
        db: instance of LofarAntennaDatabase from lofarantpos

    Example:
        >>> from lofarantpos.db import LofarAntennaDatabase
        >>> db = LofarAntennaDatabase()
        >>> pqr = get_station_pqr("DE603LBA", "outer", db)
        >>> pqr.shape
        (96, 3)
        >>> pqr[0, 0]
        1.7434713
    """
    station_type = get_station_type(station_name)

    if 'LBA' in station_name:
        # Get the PQR positions for an individual station
        station_pqr = db.antenna_pqr(station_name)

        # Exception: for Dutch stations (sparse not yet accommodated)
        if (station_type == 'core' or station_type == 'remote') and int(rcu_mode) in (3, 4):
            station_pqr = station_pqr[0:48, :]
        elif (station_type == 'core' or station_type == 'remote') and int(rcu_mode) in (1,2):
            station_pqr = station_pqr[48:, :]
    elif 'HBA' in station_name:
        selected_dipole_config = {
            'intl': GENERIC_INT_201512, 'remote': GENERIC_REMOTE_201512, 'core': GENERIC_CORE_201512
        }
        selected_dipoles = selected_dipole_config[station_type] + \
            np.arange(len(selected_dipole_config[station_type])) * 16
        station_pqr = db.hba_dipole_pqr(station_name)[selected_dipoles]
    else:
        raise RuntimeError("Station name did not contain LBA or HBA, could not load antenna positions")

    return station_pqr.astype('float32')


def make_ground_plot(image: np.ndarray, background_map: np.ndarray, extent: List[float], title: str = "Ground plot",
                     subtitle: str = "", opacity: float = 0.6, fig: Figure = None, **kwargs) \
        -> Tuple[Figure, np.ndarray]:
    """
    Make a ground plot of an array with data

    Args:
        image: numpy array (two dimensions with data)
        background_map: background map
        title: Title for the plot
        subtitle: Subtitle for the plot
        opacity: maximum opacity of the plot
        fig: exisiting figure object to be reused
        **kwargs: other options to be passed to plt.imshow (e.g. vmin)

    Returns:
        Updated figure and numpy array with only the plot

    Example:
        >>> dummy_image = np.zeros((150, 150))
        >>> fig, plot_array = make_ground_plot(dummy_image, dummy_image, [-300, 300, -100, 100])
        >>> plot_array.shape
        (150, 150, 4)
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)

    # Make colors semi-transparent in the lower 3/4 of the scale
    cmap = cm.Spectral_r
    cmap_with_alpha = cmap(np.arange(cmap.N))
    cmap_with_alpha[:, -1] = np.clip(np.linspace(0, 1.5, cmap.N), 0., 1.)
    cmap_with_alpha = ListedColormap(cmap_with_alpha)

    # Plot the resulting image
    ax = fig.add_subplot(111, ymargin=-0.4)
    ax.imshow(background_map, extent=extent)
    cimg = ax.imshow(image, origin='lower', cmap=cmap_with_alpha, extent=extent,
                     alpha=opacity, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2, axes_class=maxes.Axes)
    cbar = fig.colorbar(cimg, cax=cax, orientation="vertical", format="%.1e")
    cbar.set_alpha(1.0)
    cbar.draw_all()
    # cbar.set_ticks([])

    ax.set_xlabel('$W-E$ (metres)', fontsize=14)
    ax.set_ylabel('$S-N$ (metres)', fontsize=14)

    ax.text(0.5, 1.05, title, fontsize=17, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.5, 1.02, subtitle, fontsize=12, ha='center', va='bottom', transform=ax.transAxes)

    # Change limits to match the original specified extent in the localnorth frame
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.tick_params(axis='both', which='both', length=0)

    # Place the NSEW coordinate directions
    ax.text(0.95, 0.5, 'E', color='w', fontsize=18, transform=ax.transAxes, ha='center', va='center')
    ax.text(0.05, 0.5, 'W', color='w', fontsize=18, transform=ax.transAxes, ha='center', va='center')
    ax.text(0.5, 0.95, 'N', color='w', fontsize=18, transform=ax.transAxes, ha='center', va='center')
    ax.text(0.5, 0.05, 'S', color='w', fontsize=18, transform=ax.transAxes, ha='center', va='center')

    ground_vmin_img, ground_vmax_img = cimg.get_clim()
    ax.contour(image, np.linspace(ground_vmin_img, ground_vmax_img, 15), origin='lower', cmap=cm.Greys,
               extent=extent, linewidths=0.5, alpha=opacity)
    ax.grid(True, alpha=0.3)

    vmin, vmax = cimg.get_clim()
    raw_plotdata = cmap_with_alpha(Normalize(vmin=vmin, vmax=vmax)(image))[::-1, :]

    return fig, raw_plotdata


def make_sky_plot(image: np.ndarray, marked_bodies_lmn: Dict[str, Tuple[float, float, float]],
                  title: str = "Sky plot", subtitle: str = "", fig: Figure = None,
                  **kwargs) -> Figure:
    """
    Make a sky plot out of an array with data

    Args:
        image: numpy array (two dimensions with data)
        marked_bodies_lmn: dict with objects to annotate (values should be lmn coordinates)
        title: Title for the plot
        subtitle: Subtitle for the plot
        fig: existing figure object to be reused
        **kwargs: other options to be passed to plt.imshow (e.g. vmin)

    Returns:
        Updated figure

    Example:
        >>> dummy_image = np.zeros((150, 150))
        >>> fig = make_sky_plot(dummy_image, {})
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    circle1 = Circle((0, 0), 1.0, edgecolor='k', fill=False, facecolor='none', alpha=0.3)
    ax.add_artist(circle1)

    cimg = ax.imshow(image, origin='lower', cmap=cm.Spectral_r, extent=(1, -1, -1, 1),
                     clip_path=circle1, clip_on=True, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2, axes_class=maxes.Axes)
    fig.colorbar(cimg, cax=cax, orientation="vertical", format="%.1e")

    ax.set_xlim(1, -1)

    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Labels
    ax.set_xlabel('$ℓ$', fontsize=14)
    ax.set_ylabel('$m$', fontsize=14)

    ax.text(0.5, 1.05, title, fontsize=17, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.5, 1.02, subtitle, fontsize=12, ha='center', va='bottom', transform=ax.transAxes)

    for body_name, lmn in marked_bodies_lmn.items():
        ax.plot([lmn[0]], [lmn[1]], marker='x', color='black', mew=0.5)
        ax.annotate(body_name, (lmn[0], lmn[1]))

    # Plot the compass directions
    ax.text(0.9, 0, 'E', horizontalalignment='center', verticalalignment='center', color='w', fontsize=17)
    ax.text(-0.9, 0, 'W', horizontalalignment='center', verticalalignment='center', color='w', fontsize=17)
    ax.text(0, 0.9, 'N', horizontalalignment='center', verticalalignment='center', color='w', fontsize=17)
    ax.text(0, -0.9, 'S', horizontalalignment='center', verticalalignment='center', color='w', fontsize=17)

    return fig


def make_xst_plots(xst_filename: str,
                   station_name: str,
                   caltable_dir: str = "CalTables",
                   extent: List[float] = None,
                   pixels_per_metre: float = 0.5,
                   sky_vmin: float = None,
                   sky_vmax: float = None,
                   ground_vmin: float = None,
                   ground_vmax: float = None,
                   height: float = 1.5,
                   map_zoom: int = 19,
                   sky_only: bool = False,
                   opacity: float = 0.6):
    """
    Create sky and ground plots for an XST file

    Args:
        xst_filename: Full path to XST file
        station_name: Full station name, e.g. "DE603LBA"
        caltable_dir: Caltable directory. Defaults to "CalTables".
        extent: Extent (in m) for ground image. Defaults to [-150, 150, -150, 150]
        pixels_per_metre: Pixels per metre. Defaults to 0.5.
        height: Height (in m) for ground image. Defaults to 1.5.
        map_zoom: Zoom level for map tiles. Defaults to 19.
        sky_only: Make sky image only. Defaults to False.
        opacity: Opacity for map overlay. Defaults to 0.6.


    Returns:
        Leaflet map (that renders as an interactive map in a notebook)

    Example:
        >>> leafletmap = make_xst_plots("test/20170720_095816_mode_3_xst_sb297.dat", \
                                        "DE603LBA", caltable_dir="test/CalTables")
        Maximum at -6m east, 70m north of station center (lat/long 50.97998, 11.71118)

        >>> type(leafletmap)
        <class 'folium.folium.Map'>
    """
    cubename = os.path.basename(xst_filename)

    if extent is None:
        extent = [-150, 150, -150, 150]

    station_type = get_station_type(station_name)

    os.makedirs('results', exist_ok=True)

    # Distill metadata from filename
    obsdatestr, obstimestr, _, rcu_mode, _, subbandname = cubename.rstrip(".dat").split("_")
    subband = int(subbandname[2:])

    # Needed for NL stations: inner (rcu_mode 3/4), outer (rcu_mode 1/2), (sparse tbd)
    # Should be set to 'inner' if station type = 'intl'
    if rcu_mode in ('1', '2'):
        if len(station_name) == 5:
            station_name += "LBA"
    elif rcu_mode in ('3', '4'):
        if len(station_name) == 5:
            station_name += "LBA"
    elif rcu_mode in ('5', '6', '7'):
        if len(station_name) == 5:
            station_name += "HBA"
    else:
        raise Exception("Unexpected rcu_mode: ", rcu_mode)

    # Get the data
    fname = f"{obsdatestr}_{obstimestr}_{station_name}_SB{subband}"

    npix_l, npix_m = 131, 131
    freq = freq_from_sb(subband, rcu_mode=rcu_mode)

    # Which slice in time to visualise
    timestep = 0

    # For ground imaging
    ground_resolution = pixels_per_metre  # pixels per metre for ground_imaging, default is 0.5 pixel/metre

    obstime = datetime.datetime.strptime(obsdatestr + ":" + obstimestr, '%Y%m%d:%H%M%S')

    cube = read_acm_cube(xst_filename, station_type)

    cube, calibration_info = apply_calibration(cube, station_name, rcu_mode, subband, caltable_dir=caltable_dir)

    # Split into the XX and YY polarisations (RCUs)
    # This needs to be modified in future for LBA sparse
    cube_xx = cube[:, 0::2, 0::2]
    cube_yy = cube[:, 1::2, 1::2]
    visibilities_all = cube_xx + cube_yy

    # Stokes I for specified timestep
    visibilities = visibilities_all[timestep]

    # Setup the database
    db = LofarAntennaDatabase()

    station_pqr = get_station_pqr(station_name, rcu_mode, db)

    # Rotate station_pqr to a north-oriented xyz frame, where y points North, in a plane through the station.
    rotation = db.rotation_from_north(station_name)

    pqr_to_xyz = np.array([[np.cos(-rotation), -np.sin(-rotation), 0],
                           [np.sin(-rotation), np.cos(-rotation), 0],
                           [0, 0, 1]])

    station_xyz = (pqr_to_xyz @ station_pqr.T).T

    baselines = station_xyz[:, np.newaxis, :] - station_xyz[np.newaxis, :, :]

    # Fourier transform
    # visibilities = cube_xx[2,:,:]
    img = sky_imager(visibilities, baselines, freq, npix_l, npix_m)

    obstime_astropy = Time(obstime)
    # Determine positions of Cas A and Cyg A
    station_earthlocation = EarthLocation.from_geocentric(*(db.phase_centres[station_name] * u.m))
    zenith = AltAz(az=0 * u.deg, alt=90 * u.deg, obstime=obstime_astropy,
                   location=station_earthlocation).transform_to(GCRS)

    marked_bodies = {
        'Cas A': SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg),
        'Cyg A': SkyCoord(ra=299.868 * u.deg, dec=40.734 * u.deg),
        #        'Per A': SkyCoord.from_name("Perseus A"),
        #        'Her A': SkyCoord.from_name("Hercules A"),
        #        'Cen A': SkyCoord.from_name("Centaurus A"),
        #        '?': SkyCoord.from_name("J101415.9+105106"),
        #        '3C295': SkyCoord.from_name("3C295"),
        #        'Moon': get_moon(obstime_astropy, location=station_earthlocation).transform_to(GCRS),
        'Sun': get_sun(obstime_astropy)
        #        '3C196': SkyCoord.from_name("3C196")
    }

    marked_bodies_lmn = {}
    for body_name, body_coord in marked_bodies.items():
        # print(body_name, body_coord.separation(zenith), body_coord.separation(zenith))
        if body_coord.transform_to(AltAz(location=station_earthlocation, obstime=obstime_astropy)).alt > 0:
            marked_bodies_lmn[body_name] = skycoord_to_lmn(marked_bodies[body_name], zenith)

    # Plot the resulting sky image
    fig = plt.figure(figsize=(10, 10))

    make_sky_plot(img, marked_bodies_lmn, title=f"Sky image for {station_name}",
                  subtitle=f"SB {subband} ({freq / 1e6:.1f} MHz), {str(obstime)[:16]}", fig=fig,
                  vmin=sky_vmin, vmax=sky_vmax)

    fig.savefig(os.path.join('results', f'{fname}_sky_calibrated.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    if sky_only:
        return img

    npix_x, npix_y = int(ground_resolution * (extent[1] - extent[0])), int(ground_resolution * (extent[3] - extent[2]))

    os.environ["NUMEXPR_NUM_THREADS"] = "3"

    # Select a subset of visibilities, only the lower triangular part
    baseline_indices = np.tril_indices(visibilities.shape[0])

    visibilities_selection = visibilities[baseline_indices]

    img = nearfield_imager(visibilities_selection.flatten()[:, np.newaxis],
                           np.array(baseline_indices).T,
                           [freq], npix_x, npix_y, extent, station_xyz, height=height)

    # Correct for taking only lower triangular part
    img = np.real(2 * img)

    # Convert bottom left and upper right to PQR just for lofargeo
    pmin, qmin, _ = pqr_to_xyz.T @ (np.array([extent[0], extent[2], 0]))
    pmax, qmax, _ = pqr_to_xyz.T @ (np.array([extent[1], extent[3], 0]))
    lon_center, lat_center, _ = lofargeotiff.pqr_to_longlatheight([0, 0, 0], station_name)
    lon_min, lat_min, _ = lofargeotiff.pqr_to_longlatheight([pmin, qmin, 0], station_name)
    lon_max, lat_max, _ = lofargeotiff.pqr_to_longlatheight([pmax, qmax, 0], station_name)

    background_map = get_map(lon_min, lon_max, lat_min, lat_max, zoom=map_zoom)

    fig, folium_overlay = make_ground_plot(img, background_map, extent,
                                           title=f"Near field image for {station_name}",
                                           subtitle=f"SB {subband} ({freq / 1e6:.1f} MHz), {str(obstime)[:16]}",
                                           opacity=opacity, vmin=ground_vmin, vmax=ground_vmax)

    fig.savefig(os.path.join("results", f"{fname}_nearfield_calibrated.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)

    maxpixel_ypix, maxpixel_xpix = np.unravel_index(np.argmax(img), img.shape)
    maxpixel_x = np.interp(maxpixel_xpix, [0, npix_x], [extent[0], extent[1]])
    maxpixel_y = np.interp(maxpixel_ypix, [0, npix_y], [extent[2], extent[3]])
    [maxpixel_p, maxpixel_q, _] = pqr_to_xyz.T @ np.array([maxpixel_x, maxpixel_y, height])
    maxpixel_lon, maxpixel_lat, _ = lofargeotiff.pqr_to_longlatheight([maxpixel_p, maxpixel_q], station_name)

    # Show location of maximum if not at the image border
    if 2 < maxpixel_xpix < npix_x - 2 and 2 < maxpixel_ypix < npix_y - 2:
        print(f"Maximum at {maxpixel_x:.0f}m east, {maxpixel_y:.0f}m north of station center " +
              f"(lat/long {maxpixel_lat:.5f}, {maxpixel_lon:.5f})")

    obstime = datetime.datetime.strptime(obsdatestr + ":" + obstimestr, '%Y%m%d:%H%M%S')

    tags = {"datafile": xst_filename,
            "generated_with": f"lofarimaging v{__version__}",
            "subband": subband,
            "frequency": freq,
            "extent_xyz": extent,
            "height": height,
            "station": station_name,
            "pixels_per_metre": pixels_per_metre}
    tags.update(calibration_info)
    lofargeotiff.write_geotiff(img, os.path.join("results", f"{fname}_nearfield_calibrated.tiff"),
                               (pmin, qmin), (pmax, qmax), stationname=station_name,
                               obsdate=obstime, tags=tags)

    return make_leaflet_map(folium_overlay, lon_center, lat_center, lon_min, lat_min, lon_max, lat_max)
