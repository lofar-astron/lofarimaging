"""Functions for working with LOFAR single station data"""

from typing import Dict, List
import numpy as np
from numpy.linalg import norm, lstsq
import numexpr as ne
import numba
from astropy.coordinates import SkyCoord, SkyOffsetFrame, CartesianRepresentation


__all__ = ["nearfield_imager", "sky_imager", "numba_sky_imager", "ground_imager", "skycoord_to_lmn", "calibrate", "simulate_sky_source",
           "subtract_sources"]

__version__ = "1.5.0"
SPEED_OF_LIGHT = 299792458.0


def skycoord_to_lmn(pos: SkyCoord, phasecentre: SkyCoord):
    """
    Convert astropy sky coordinates into the l,m,n coordinate system
    relative to a phase centre.

    The l,m,n is a RHS coordinate system with
    * its origin on the sky sphere
    * m,n and the celestial north on the same plane
    * l,m a tangential plane of the sky sphere

    Note that this means that l increases east-wards

    This function was taken from https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library
    """

    # Determine relative sky position
    todc = pos.transform_to(SkyOffsetFrame(origin=phasecentre))
    dc = todc.represent_as(CartesianRepresentation)
    dc /= dc.norm()

    # Do coordinate transformation - astropy's relative coordinates do
    # not quite follow imaging conventions
    return dc.y.value, dc.z.value, dc.x.value - 1


def sky_imager(visibilities, baselines, freq, npix_l, npix_m):
    """
    Sky imager

    Args:
        visibilities: Numpy array with visibilities, shape [num_antennas x num_antennas]
        baselines: Numpy array with distances between antennas, shape [num_antennas, num_antennas, 3]
        freq: frequency
        npix_l: Number of pixels in l-direction
        npix_m: Number of pixels in m-direction

    Returns:
        np.array(float): Real valued array of shape [npix_l, npix_m]
    """
    img = np.zeros((npix_m, npix_l), dtype=np.complex128)

    for m_ix in range(npix_m):
        m = -1 + m_ix * 2 / npix_m
        for l_ix in range(npix_l):
            l = 1 - l_ix * 2 / npix_l
            img[m_ix, l_ix] = np.mean(visibilities * np.exp(-2j * np.pi * freq *
                                                            (baselines[:, :, 0] * l + baselines[:, :, 1] * m) /
                                                            SPEED_OF_LIGHT))
    return np.real(img)


@numba.jit(parallel=True)
def numba_sky_imager(visibilities, baselines, freq, npix_l, npix_m):
    """
    Sky imager

    Args:
        visibilities: Numpy array with visibilities, shape [num_antennas x num_antennas]
        baselines: Numpy array with distances between antennas, shape [num_antennas, num_antennas, 3]
        freq: frequency
        npix_l: Number of pixels in l-direction
        npix_m: Number of pixels in m-direction

    Returns:
        np.array(float): Real valued array of shape [npix_l, npix_m]
    """
    img = np.zeros((npix_m, npix_l), dtype=np.complex128)

    for m_ix in range(npix_m):
        m = -1 + m_ix * 2 / npix_m
        for l_ix in range(npix_l):
            l = 1 - l_ix * 2 / npix_l
            img[m_ix, l_ix] = np.mean(visibilities * np.exp(-2j * np.pi * freq *
                                                            (baselines[:, :, 0] * l + baselines[:, :, 1] * m) /
                                                            SPEED_OF_LIGHT))
    return np.real(img)


def ground_imager(visibilities, freq, npix_p, npix_q, dims, station_pqr, height=1.5):
    """Do a Fourier transform for ground imaging"""
    img = np.zeros([npix_q, npix_p], dtype=np.complex128)

    for q_ix, q in enumerate(np.linspace(dims[2], dims[3], npix_q)):
        for p_ix, p in enumerate(np.linspace(dims[0], dims[1], npix_p)):
            r = height
            pqr = np.array([p, q, r], dtype=np.float32)
            antdist = np.linalg.norm(station_pqr - pqr[np.newaxis, :], axis=1)
            groundbase = antdist[:, np.newaxis] - antdist[np.newaxis, :]
            img[q_ix, p_ix] = np.mean(visibilities * np.exp(-2j * np.pi * freq * (-groundbase) / SPEED_OF_LIGHT))

    return img


def nearfield_imager(visibilities, baseline_indices, freqs, npix_p, npix_q, extent, station_pqr, height=1.5,
                     max_memory_mb=200):
    """
    Nearfield imager

    Args:
        visibilities: Numpy array with visibilities, shape [num_visibilities x num_frequencies]
        baseline_indices: List with tuples of antenna numbers in visibilities, shape [2 x num_visibilities]
        freqs: List of frequencies
        npix_p: Number of pixels in p-direction
        npix_q: Number of pixels in q-direction
        extent: Extent (in m) that the image should span
        station_pqr: PQR coordinates of stations
        height: Height of image in metre
        max_memory_mb: Maximum amount of memory to use for the biggest array. Higher may improve performance.

    Returns:
        np.array(complex): Complex valued array of shape [npix_p, npix_q]
    """
    z = height
    x = np.linspace(extent[0], extent[1], npix_p)
    y = np.linspace(extent[2], extent[3], npix_q)

    posx, posy = np.meshgrid(x, y)
    posxyz = np.transpose(np.array([posx, posy, z * np.ones_like(posx)]), [1, 2, 0])

    diff_vectors = (station_pqr[:, None, None, :] - posxyz[None, :, :, :])
    distances = np.linalg.norm(diff_vectors, axis=3)

    vis_chunksize = max_memory_mb * 1024 * 1024 // (8 * npix_p * npix_q)

    bl_diff = np.zeros((vis_chunksize, npix_q, npix_p), dtype=np.float64)
    img = np.zeros((npix_q, npix_p), dtype=np.complex128)
    for vis_chunkstart in range(0, len(baseline_indices), vis_chunksize):
        vis_chunkend = min(vis_chunkstart + vis_chunksize, baseline_indices.shape[0])
        # For the last chunk, bl_diff_chunk is a bit smaller than bl_diff
        bl_diff_chunk = bl_diff[:vis_chunkend - vis_chunkstart, :]
        np.add(distances[baseline_indices[vis_chunkstart:vis_chunkend, 0]],
               -distances[baseline_indices[vis_chunkstart:vis_chunkend, 1]], out=bl_diff_chunk)

        j2pi = 1j * 2 * np.pi
        for ifreq, freq in enumerate(freqs):
            v = visibilities[vis_chunkstart:vis_chunkend, ifreq][:, None, None]
            lamb = SPEED_OF_LIGHT / freq

            # v[:,np.newaxis,np.newaxis]*np.exp(-2j*np.pi*freq/c*groundbase_pixels[:,:,:]/c)
            # groundbase_pixels=nvis x npix x npix
            np.add(img, np.sum(ne.evaluate("v * exp(j2pi * bl_diff_chunk / lamb)"), axis=0), out=img)
    img /= len(freqs) * len(baseline_indices)

    return img


def calibrate(vis, modelvis, maxiter=30, amplitudeonly=True):
    """
    Calibrate and subtract some sources

    Args:
        vis: visibility matrix, shape [n_st, n_st]
        modelvis: model visibility matrices, shape [n_dir, n_st, n_st]
        maxiter: max iterations (default 30)
        amplitudeonly: fit only amplitudes (default True)

    Returns:
        residual: visibilities with calibrated directions subtracted, shape [n_st, n_st]
        gains: gains, shape [n_dir, n_st]
    """
    nst = vis.shape[1]
    ndir = np.array(modelvis).shape[0]
    gains = np.ones([ndir, nst], dtype=np.complex)

    if ndir == 0:
        return vis, gains
    else:
         gains *= np.sqrt(norm(vis) / norm(modelvis))

    iteration = 0
    while iteration < maxiter:
        iteration += 1
        gains_prev = gains.copy()
        for k in range(nst):
            z = np.conj(gains_prev) * np.array(modelvis)[:, :, k]
            gains[:, k] = lstsq(z.T, vis[:, k], rcond=None)[0]
        if amplitudeonly:
            gains = np.abs(gains).astype(np.complex)
        if iteration % 2 == 0 and iteration > 0:
            dg = norm(gains - gains_prev)
            residual = vis.copy()
            for d in range(ndir):
                residual -= np.diag(np.conj(gains[d])) @ modelvis[d] @ np.diag(gains[d])
            gains = 0.5 * gains + 0.5 * gains_prev
    return residual, gains


def simulate_sky_source(lmn_coord: np.array, baselines: np.array, freq: float):
    """
    Simulate visibilities for a sky source

    Args:
        lmn_coord (np.array): l, m, n coordinate
        baselines (np.array): baseline distances in metres, shape (n_ant, n_ant)
        freq (float): Frequency in Hz
    """
    return np.exp(2j * np.pi * freq * baselines.dot(np.array(lmn_coord)) / SPEED_OF_LIGHT)


def subtract_sources(vis: np.array, baselines: np.array, freq: float, lmn_dict: Dict[str, np.array],
                     sources=["Cas A", "Cyg A", "Sun"]):
    """
    Subtract sky sources from visibilities

    Args:
        vis (np.array): visibility matrix, shape [n_ant, n_ant]
        lmn_dict (Dict[str, np.array]): dictionary with lmn coordinates
        baselines (np.array): baseline distances in metres, shape (n_ant, n_ant)
        freq (float): Frequency in Hz
        sources (List[str]): list with source names to subtract (should all be in lmn_dict).
                             Default ["Cas A", "Sun"]

    Returns:
        vis (np.array): visibility matrix with sources subtracted
    """
    modelvis = [simulate_sky_source(lmn_dict[srcname], baselines, freq) for srcname in lmn_dict
                if srcname in sources]

    residual, _ = calibrate(vis, modelvis)

    return residual
