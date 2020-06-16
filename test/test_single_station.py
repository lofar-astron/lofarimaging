import numpy
from numpy.testing import assert_almost_equal

from lofarimaging.lofarimaging import compute_calibrated_model, \
    compute_pointing_matrix, simulate_sky_source, \
    estimate_sources_flux, \
    estimate_model_visibilities, \
    self_cal


def test_compute_calibrated_model():
    # ----TEST DATA
    vis = numpy.diag(numpy.ones((10)))
    model_vis = numpy.diag(numpy.ones(10))
    # ----TEST
    calibrated, gains, residual = compute_calibrated_model(vis, model_vis)
    assert_almost_equal(calibrated, vis)
    assert residual < 1.e-5


def test_simulate_sky_source():
    # ----TEST DATA
    source_position = [0, 0, 0]
    baselines = numpy.zeros((2, 2, 3))
    baselines[0, 1, :] = [1, 0, 0]
    baselines[1, 0, :] = [1, 0, 0]
    # ----TEST
    visibilities = simulate_sky_source(source_position, baselines, 1)
    expected = numpy.ones((2, 2))
    assert_almost_equal(visibilities, expected)


def test_compute_pointing_matrix():
    # ----TEST DATA
    source_position = numpy.zeros((1, 3))
    baselines = numpy.zeros((2, 2, 3))
    baselines[0, 1, :] = [1, 0, 0]
    baselines[1, 0, :] = [1, 0, 0]
    expected = numpy.ones((2, 2, 1))
    # ----TEST
    pointing_matrix = compute_pointing_matrix(source_position, baselines, 1)
    assert_almost_equal(pointing_matrix, expected)


def test_estimate_sources_flux():
    # ----TEST DATA
    source_position = numpy.zeros((1, 3))
    pointing_matrix = numpy.ones((2, 2, 1))
    visibilities = numpy.ones((2, 2))
    # ----TEST
    source_fluxes = estimate_sources_flux(visibilities, pointing_matrix)
    expected = numpy.ones((1))
    assert_almost_equal(source_fluxes, expected)


def test_estimate_model_visibilities():
    # ----TEST DATA
    source_position = numpy.zeros((1, 3))
    visibilities = numpy.ones((2, 2))
    baselines = numpy.zeros((2, 2, 3))
    baselines[0, 1, :] = [1, 0, 0]
    baselines[1, 0, :] = [1, 0, 0]
    # ----TEST
    model_visibilities = estimate_model_visibilities(source_position, visibilities, baselines, 1)
    assert_almost_equal(visibilities, model_visibilities)


def test_self_cal_zero_residual():
    # ----TEST DATA
    source_position = numpy.zeros((1, 3))
    visibilities = numpy.ones((2, 2))
    baselines = numpy.zeros((2, 2, 3))
    baselines[0, 1, :] = [1, 0, 0]
    baselines[1, 0, :] = [1, 0, 0]
    # ----TEST
    calibrated_model, _ = self_cal(visibilities, source_position, baselines, 1)
    assert_almost_equal(calibrated_model, visibilities)
