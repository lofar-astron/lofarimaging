"""Functions for working with LOFAR single station data"""

__all__ = ["sb_from_freq", "freq_from_sb", "find_caltable", "read_caltable",
           "rcus_in_station", "read_acm_cube", "get_background_image",
           "sky_imager", "ground_imager", "get_extents_pqr"]

import numpy as np
import os
from matplotlib.pyplot import imread
import warnings


def sb_from_freq(freq: float, clock=200.e6):
    """
    Convert subband number to central frequency
    
    Args:
        freq: frequency in Hz
        clock: clock speed in Hz

    Returns:
        int: subband number
    """
    chan = 0.5 * clock / 512.
    sb = round(freq / chan)
    return int(sb)


def freq_from_sb(sb: int, clock=200e6):
    """
    Convert central frequency to subband number

    Args:
        sb: subband number
        clock: clock speed in Hz

    Returns:
        float: frequency in Hz
    """
    chan = 0.5 * clock / 512.
    freq = (sb * chan)
    return freq


def find_caltable(field_name: str, rcu_mode: int, config_dir='caltables'):
    """
    Find the file of a caltable.

    Args:
        field_name: Name of the antenna field, e.g. 'DE602LBA'
        rcu_mode: Receiver mode for which the calibration table is requested.
            An integer from 1 to 7 inclusive.
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
    else:
        filename = os.path.join(config_dir, f"CalTable_{station_number}_mode{rcu_mode}.dat")

    if os.path.exists(filename):
        return filename

    # If the original folder structure is kept
    if rcu_mode == 'outer' and 'LBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-LBA_OUTER-10_90.dat")
    elif rcu_mode == 'inner' and 'LBA' in field_name:
        filename = os.path.join(config_dir, f"{station}/CalTable-{station_number}-LBA_INNER-10_90.dat")
    else:
        filename = os.path.join(config_dir, f"{station}/CalTable_{station_number}_mode{rcu_mode}.dat")

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
        Tuple(List[str], np.array): A tuple containing a list of strings with
            the header lines, and a 2D numpy.array of complex numbers
            representing the station gain coefficients.
    """
    infile = open(filename, 'rb')

    header_lines = []

    try:
        while True:
            header_lines.append(infile.readline().decode('utf8'))
            if 'HeaderStop' in header_lines[-1]:
                break
    except UnicodeDecodeError:
        # No header; close and open again
        infile.close()
        infile = open(filename, 'rb')

    caldata = np.fromfile(infile, dtype=np.complex128)
    num_rcus = len(caldata) // num_subbands

    infile.close()

    return header_lines, caldata.reshape((num_subbands, num_rcus))


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


def get_background_image(lon_min, lon_max, lat_min, lat_max, zoom=19):
    """
    Get an ESRI World Imagery map of the selected region
    Args:
        lon_min: Minimum longitude (degrees)
        lon_max: Maximum longitude (degrees)
        lat_min: Minimum latitude (degrees)
        lat_max: Maximum latitude (degrees)
        zoom: Zoom level

    Returns:
        np.array: Numpy array which can be plotted with plt.imshow
    """
    from owslib.wmts import WebMapTileService
    import mercantile

    wmts = WebMapTileService("http://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml")

    upperleft_tile = mercantile.tile(lon_min, lat_max, zoom)
    xmin, ymin = upperleft_tile.x, upperleft_tile.y
    lowerright_tile = mercantile.tile(lon_max, lat_min, zoom)
    xmax, ymax = lowerright_tile.x, lowerright_tile.y

    total_image = np.zeros([256 * (ymax - ymin + 1), 256 * (xmax - xmin + 1), 3], dtype='uint8')

    tile_min = mercantile.tile(lon_min, lat_min, zoom)
    tile_max = mercantile.tile(lon_max, lat_max, zoom)

    for x in range(tile_min.x, tile_max.x + 1):
        for y in range(tile_max.y, tile_min.y + 1):
            tile = wmts.gettile(layer="World_Imagery", tilematrix=str(zoom), row=y, column=x)
            out = open("tmp.jpg", "wb")
            out.write(tile.read())
            out.close()
            tile_image = imread("tmp.jpg")
            total_image[(y - ymin) * 256: (y - ymin + 1) * 256,
                        (x - xmin) * 256: (x - xmin + 1) * 256] = tile_image

    total_lonlatmin = {'lon': mercantile.bounds(xmin, ymax, zoom).west, 'lat': mercantile.bounds(xmin, ymax, zoom).south}
    total_lonlatmax = {'lon': mercantile.bounds(xmax, ymin, zoom).east, 'lat': mercantile.bounds(xmax, ymin, zoom).north}

    pix_xmin = int(round(np.interp(lon_min, [total_lonlatmin['lon'], total_lonlatmax['lon']], [0, total_image.shape[1]])))
    pix_ymin = int(round(np.interp(lat_min, [total_lonlatmin['lat'], total_lonlatmax['lat']], [0, total_image.shape[0]])))
    pix_xmax = int(round(np.interp(lon_max, [total_lonlatmin['lon'], total_lonlatmax['lon']], [0, total_image.shape[1]])))
    pix_ymax = int(round(np.interp(lat_max, [total_lonlatmin['lat'], total_lonlatmax['lat']], [0, total_image.shape[0]])))
    return total_image[total_image.shape[0]-pix_ymax: total_image.shape[0]-pix_ymin, pix_xmin: pix_xmax]


SPEED_OF_LIGHT = 299792458.0


def sky_imager(visibilities, baselines, freq, im_x, im_y):
    """Do a Fourier transform for sky imaging"""
    img = np.zeros([im_y, im_x], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
        for m_ix, m in enumerate(np.linspace(-1, 1, im_x)):
            for l_ix, l in enumerate(np.linspace(1, -1, im_y)):
                img[m_ix, l_ix] = np.mean(visibilities *
                                          np.exp(-2j * np.pi * freq *
                                                 (baselines[:, :, 0] * l + baselines[:, :, 1] * m) / SPEED_OF_LIGHT))
    return img


def ground_imager(visibilities, baselines, freq, im_x, im_y, dims, station_pqr, height=1.5):
    """Do a Fourier transform for ground imaging"""
    img = np.zeros([im_y, im_x], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
        for q_ix, q in enumerate(np.linspace(dims[2], dims[3], im_y)):
            for p_ix, p in enumerate(np.linspace(dims[0], dims[1], im_x)):
                r = height
                pqr = np.array([p, q, r], dtype=np.float32)
                antdist = np.linalg.norm(station_pqr - pqr[np.newaxis, :], axis=1)
                groundbase = antdist[:, np.newaxis] - antdist[np.newaxis, :]
                # Note: this is RFI integration second - normal second, to take out interference
                img[q_ix, p_ix] = np.mean(visibilities * np.exp(-2j * np.pi * freq * (-groundbase) / SPEED_OF_LIGHT))
    return img


def get_extents_pqr(rot_matrix, extents_localnorth, margin=5):
    """
    Get the extents of a rectangular grid in the PQR frame which contains the
    entire extents in the localnorth frame.
    A bit of margin is taken to accomodate for interpolation after rotation.

    Args:
        rot_matrix: rotation matrix from PQ to XY
        extents_localnorth: extents in the form [xmin, xmax, ymin, ymax]
        margin: pixels to add to accomodate for interpolation after rotation

    Returns:
        extents in the PQR frame, in the form [pmin, pmax, qmin, qmax]
    """
    [xmin, xmax, ymin, ymax] = extents_localnorth
    pmin2, _ = rot_matrix @ [xmin, ymin]
    _, qmin2 = rot_matrix @ [xmax, ymin]
    pmax2, _ = rot_matrix @ [xmax, ymax]
    _, qmax2 = rot_matrix @ [xmin, ymax]
    return [pmin2 - margin, pmax2 + margin, qmin2 - margin, qmax2 + margin]
