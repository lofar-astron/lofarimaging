"""Functions for working with LOFAR single station data"""

__all__ = ["sb_from_freq", "freq_from_sb", "find_caltable", "read_caltable",
           "rcus_in_station", "read_acm_cube"]

import numpy as np
import os


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

