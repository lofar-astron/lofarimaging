"""Functions for working with LOFAR single station data"""

import numpy as np
import os
import datetime
import lofargeotiff

from lofarantpos.db import LofarAntennaDatabase

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
import matplotlib.axes as maxes

from astropy.coordinates import SkyCoord, GCRS, EarthLocation, AltAz, get_sun
import astropy.units as u
from astropy.time import Time

from typing import List, Dict, Any

import lofarantpos
from packaging import version

from .maputil import get_map, make_leaflet_map
from .lofarimaging import nearfield_imager, sky_imager, skycoord_to_lmn


__all__ = ["sb_from_freq", "freq_from_sb", "find_caltable", "read_caltable",
           "rcus_in_station", "read_acm_cube", "get_station_pqr",
           "make_ground_image", "make_sky_plot"]

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


def sb_from_freq(freq: float, rcu_mode='1'):
    """
    Convert subband number to central frequency

    Args:
        rcu_mode: rcu mode
        freq: frequency in Hz

    Returns:
        int: subband number
    """
    clock = 200e6
    if rcu_mode == '6':
        clock = 160e6
    sb_bandwidth = 0.5 * clock / 512.
    freq_offset = 0
    if rcu_mode == '5':
        freq_offset = 100e6
    elif rcu_mode == '6':
        freq_offset = 160e6
    elif rcu_mode == '7':
        freq_offset = 200e6
    sb = round((freq - freq_offset) / sb_bandwidth)
    return int(sb)


def freq_from_sb(sb: int, rcu_mode='1'):
    """
    Convert central frequency to subband number

    Args:
        rcu_mode: rcu mode
        sb: subband number

    Returns:
        float: frequency in Hz
    """
    clock = 200e6
    if rcu_mode == '6':
        clock = 160e6
    freq_offset = 0
    if rcu_mode == '5':
        freq_offset = 100e6
    elif rcu_mode == '6':
        freq_offset = 160e6
    elif rcu_mode == '7':
        freq_offset = 200e6
    sb_bandwidth = 0.5 * clock / 512.
    freq = (sb * sb_bandwidth) + freq_offset
    return freq


def find_caltable(field_name: str, rcu_mode: str, config_dir='caltables'):
    """
    Find the file of a caltable.
    Args:
        field_name: Name of the antenna field, e.g. 'DE602LBA'
        rcu_mode: Receiver mode for which the calibration table is requested.
            Probably should be  'inner' or 'outer'
        config_dir: Root directory under which station information is stored in
            subdirectories DE602C/etc/, RS106/etc/, ...
    Returns:
        str: filename if it exists, None if nothing found
    """
    station, field = field_name[0:5].upper(), field_name[5:].upper()
    station_number = station[2:5]
    # Map to the correct file depending on the RCU mode
    if rcu_mode == 'outer' and 'LBA' in field_name:
        filename = os.path.join(config_dir, f"CalTable-{station_number}-LBA_OUTER-10_90.dat")
    elif rcu_mode == 'inner' and 'LBA' in field_name:
        filename = os.path.join(config_dir, f"CalTable-{station_number}-LBA_INNER-10_90.dat")
    elif rcu_mode == '5' and 'HBA' in field_name:
        filename = os.path.join(config_dir, f"CalTable-{station_number}-HBA-110_190.dat")
    elif rcu_mode == '6' and 'HBA' in field_name:
        filename = os.path.join(config_dir, f"CalTable-{station_number}-HBA-170_230.dat")
    elif rcu_mode == '7' and 'HBA' in field_name:
        filename = os.path.join(config_dir, f"CalTable-{station_number}-HBA-210_250.dat")

    if os.path.exists(filename):
        return filename
    # If the original folder structure is kept
    if rcu_mode == 'outer' and 'LBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-LBA_OUTER-10_90.dat")
    elif rcu_mode == 'inner' and 'LBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-LBA_INNER-10_90.dat")
    elif rcu_mode == '5' and 'HBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-HBA-110_190.dat")
    elif rcu_mode == '6' and 'HBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-HBA-170_230.dat")
    elif rcu_mode == '7' and 'HBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-HBA-210_250.dat")

    if os.path.exists(filename):
        return filename
    else:
        return None


def read_caltable(filename: str, num_subbands=512):
    """
    Read a station's calibration table.

    Args:
        filename: Filename with the caltable
        num_subbands: Number of subbands

    Returns:
        Tuple(Dict[str, str], np.array): A tuple containing a dict with
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


def rcus_in_station(station_type: str):
    """
    Give the number of RCUs in a station, given its type.

    Args:
        station_type: Kind of station that produced the correlation. One of
            'core', 'remote', 'intl'.
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

    Examples:
    >>> cube = read_acm_cube('20170720_095816_xst.dat', 'intl')
    >>> cube.shape
    (29, 192, 192)
    """
    num_rcu = rcus_in_station(station_type)
    data = np.fromfile(filename, dtype=np.complex128)
    time_slots = int(len(data) / num_rcu / num_rcu)
    return data.reshape((time_slots, num_rcu, num_rcu))


def get_station_pqr(station_name: str, station_type: str, array_type: str, db):
    if 'LBA' in station_name:
        # Get the PQR positions for an individual station
        station_pqr = db.antenna_pqr(station_name)

        # Exception: for Dutch stations (sparse not yet accommodated)
        if (station_type == 'core' or station_type == 'remote') and array_type == 'inner':
            station_pqr = station_pqr[0:48, :]
        elif (station_type == 'core' or station_type == 'remote') and array_type == 'outer':
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


def make_sky_plot(image: np.array, marked_bodies_lmn: Dict[str, Any],
                  title: str = "Sky plot", subtitle: str = "", fig: plt.Figure = None,
                  **kwargs):
    """

    Args:
        image: numpy array (two dimensions with data)
        marked_bodies_lmn: dict with objects to annotate (values should be lmn coordinates)
        title: Title for the plot
        subtitle: Subtitle for the plot
        fig: existing figure object to be reused
        **kwargs: other options to be passed to plt.imshow (e.g. vmin)

    Returns:
        list of matplotlib added to the axes
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


def make_ground_image(xst_filename,
                      station_name,
                      caltable_dir,
                      extent=None,
                      pixels_per_metre=0.5,
                      sky_vmin=None,
                      sky_vmax=None,
                      ground_vmin=None,
                      ground_vmax=None,
                      height=1.5,
                      map_zoom=19,
                      sky_only=False,
                      opacity=0.6):
    """Make a ground image"""
    cubename = os.path.basename(xst_filename)

    if extent is None:
        extent = [-150, 150, -150, 150]

    if station_name[0] == "C":
        station_type = "core"
    elif station_name[0] == "R" or station_name[:5] == "PL611":
        station_type = "remote"
    else:
        station_type = "intl"

    os.makedirs('results', exist_ok=True)

    # Distill metadata from filename
    obsdatestr, obstimestr, _, rcu_mode, _, subbandname = cubename.rstrip(".dat").split("_")
    subband = int(subbandname[2:])

    # Needed for NL stations: inner (rcu_mode 3/4), outer (rcu_mode 1/2), (sparse tbd)
    # Should be set to 'inner' if station type = 'intl'
    array_type = None
    if rcu_mode in ('1', '2'):
        array_type = 'outer'
    elif rcu_mode in ('3', '4'):
        array_type = 'inner'
    elif rcu_mode in ('5', '6', '7'):
        array_type = rcu_mode
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

    # Apply calibration

    caltable_filename = find_caltable(station_name, rcu_mode=array_type,
                                      config_dir=caltable_dir)

    cal_header = None
    if caltable_filename is None:
        print('No calibration table found... cube remains uncalibrated!')
    else:
        cal_header, cal_data = read_caltable(caltable_filename)

        rcu_gains = cal_data[subband, :]
        rcu_gains = np.array(rcu_gains, dtype=np.complex64)
        gain_matrix = rcu_gains[np.newaxis, :] * np.conj(rcu_gains[:, np.newaxis])
        cube = cube / gain_matrix

    # Split into the XX and YY polarisations (RCUs)
    # This needs to be modified in future for LBA sparse
    cube_xx = cube[:, 0::2, 0::2]
    cube_yy = cube[:, 1::2, 1::2]
    visibilities_all = cube_xx + cube_yy

    # Stokes I for specified timestep
    visibilities = visibilities_all[timestep]

    # Setup the database
    db = LofarAntennaDatabase()

    station_pqr = get_station_pqr(station_name, station_type, array_type, db)

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
                  subtitle=f"SB {subband} ({freq / 1e6:.1f} MHz), {str(obstime)[:16]}", fig=fig)

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

    # Make colors semi-transparent in the lower 3/4 of the scale
    cmap = cm.Spectral_r
    cmap_with_alpha = cmap(np.arange(cmap.N))
    cmap_with_alpha[:, -1] = np.clip(np.linspace(0, 1.5, cmap.N), 0., 1.)
    cmap_with_alpha = ListedColormap(cmap_with_alpha)

    # Plot the resulting image
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_subplot(111, ymargin=-0.4)
    ax.imshow(background_map, extent=extent)
    cimg = ax.imshow(img, origin='lower', cmap=cmap_with_alpha, extent=extent,
                     alpha=0.7, vmin=ground_vmin, vmax=ground_vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2, axes_class=maxes.Axes)
    cbar = fig.colorbar(cimg, cax=cax, orientation="vertical", format="%.1e")
    cbar.set_alpha(1.0)
    cbar.draw_all()
    # cbar.set_ticks([])

    ax.set_xlabel('$W-E$ (metres)', fontsize=14)
    ax.set_ylabel('$S-N$ (metres)', fontsize=14)

    ax.text(0.5, 1.05, f"Near field image for {station_name}",
            fontsize=17, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.5, 1.02, f"SB {subband} ({freq / 1e6:.1f} MHz), {str(obstime)[:16]}",
            fontsize=12, ha='center', va='bottom', transform=ax.transAxes)

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
    ax.contour(img, np.linspace(ground_vmin_img, ground_vmax_img, 15), origin='lower', cmap=cm.Greys,
               extent=extent, linewidths=0.5, alpha=opacity)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join("results", f"{fname}_nearfield_calibrated.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)

    vmin, vmax = cimg.get_clim()
    folium_overlay = cmap_with_alpha(Normalize(vmin=vmin, vmax=vmax)(img))[::-1, :]

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
            "caltable": caltable_filename,
            "subband": subband,
            "frequency": freq,
            "extent_xyz": extent,
            "height": height,
            "station": station_name,
            "pixels_per_metre": pixels_per_metre}
    if cal_header is not None:
        if "CalTableHeader.Observation.Date" in cal_header:
            tags["calibration_obsdate"] = cal_header["CalTableHeader.Observation.Date"]
        if "CalTableHeader.Calibration.Date" in cal_header:
            tags["calibration_date"] = cal_header["CalTableHeader.Calibration.Date"]
        if "CalTableHeader.Comment" in cal_header:
            tags["calibration_comment"] = cal_header["CalTableHeader.Comment"]
    lofargeotiff.write_geotiff(img, os.path.join("results", f"{fname}_nearfield_calibrated.tiff"),
                               (pmin, qmin), (pmax, qmax), stationname=station_name,
                               obsdate=obstime, tags=tags)

    return make_leaflet_map(folium_overlay, lon_center, lat_center, lon_min, lat_min, lon_max, lat_max)
