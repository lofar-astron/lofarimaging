import h5py
import numpy
import lofarimaging.hdf5util as h5utils
from lofarimaging.singlestationutil import compute_baselines, make_sky_plot
from lofarimaging.lofarimaging import compute_pointing_matrix, compute_calibrated_model, sky_imager,\
    self_cal, estimate_model_visibilities
import matplotlib.pyplot as plt

test_data_path = '/home/mmancini/Documents/Projects/SingleStationImager/test_dataset.h5'
test_station = 'CS103'
test_data = h5py.File(test_data_path, 'r')
obs_names = h5utils.get_obsnums(test_data, station_name=test_station)
an_obs = test_data[obs_names[10]]

a_frequency = an_obs.attrs['frequency']
a_source_lmn = an_obs.attrs['source_lmn']
a_source_name = an_obs.attrs['source_names']
a_rcu_mode = an_obs.attrs['rcu_mode']
a_dataset = an_obs['calibrated_data']

a_dataset_xx = a_dataset[::2, ::2]
a_dataset_yy = a_dataset[1::2, 1::2]

a_dataset_i = a_dataset_xx + a_dataset_yy
a_dataset_q = a_dataset_xx - a_dataset_yy
a_dataset_u = 2 * (a_dataset_xx * a_dataset_yy.conj()).real
a_dataset_v = -2 * (a_dataset_xx * a_dataset_yy.conj()).imag

baselines = compute_baselines(test_station, a_rcu_mode)
sky_image = sky_imager(a_dataset_i, baselines, a_frequency, 100, 100)

model = estimate_model_visibilities(a_source_lmn, a_dataset_i, baselines, a_frequency)

model_image = sky_imager(model, baselines, a_frequency, 100, 100)

pointing_matrix = compute_pointing_matrix(a_source_lmn, baselines, a_frequency)
calibrated_model, _ = compute_calibrated_model(a_dataset_i, model, maxiter=50)
self_cal_model = self_cal(a_dataset_i, a_source_lmn, baselines, a_frequency)

sky_without_calibrated_model = sky_imager(a_dataset_i - self_cal_model, baselines, a_frequency, 100, 100)
calibrated_model_image = sky_imager(calibrated_model, baselines, a_frequency, 100, 100)
sky_without_model = sky_imager(a_dataset_i - model, baselines, a_frequency, 100, 100)


f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
min, max = numpy.min(sky_image), numpy.max(sky_image)

ax1.scatter(a_source_lmn[:, 0], a_source_lmn[:, 1], marker='o', color='r', s=100, facecolors='none')
im = ax1.imshow(sky_image[::-1,:], vmin=min, vmax=max, extent=(1, -1, -1, 1))
ax1.set_title('initial sky')

ax2.scatter(a_source_lmn[:, 0], a_source_lmn[:, 1], marker='o', color='r', s=100, facecolors='none')
ax2.imshow(sky_without_model[::-1, :], extent=(1, -1, -1, 1), vmin=min, vmax=max)
ax2.set_title('sky without model')

ax3.scatter(a_source_lmn[:, 0], a_source_lmn[:, 1], marker='o', color='r', s=100, facecolors='none')
ax3.imshow(calibrated_model_image[::-1, :], extent=(1, -1, -1, 1))
ax3.set_title('calibrated model image')

ax4.imshow(sky_without_calibrated_model[::-1, :], extent=(1, -1, -1, 1), vmin=min, vmax=max)
to_plot = {name: pos for pos, name in zip(a_source_lmn, a_source_name)}
ax4.set_title('sky without calibrated model')

make_sky_plot(sky_image, to_plot)

plt.show()
