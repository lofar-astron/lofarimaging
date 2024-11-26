#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")
import time
from lofarimaging.opc_interface import download_xst
from lofarimaging import make_xst_plots
import threading
import matplotlib.pyplot as plt
import logging

integration_time_s = 5

logger = logging.getLogger(__name__)

def download_and_image(station_name, subband):
    port = 50000 + int(station_name[2:])
    logger.info("Requesting data for station {} at subband {}".format(station_name, subband))
    obstime, visibilities, rcu_mode = download_xst(subband, integration_time_s, port=port)
    if rcu_mode in (1, 2, 3, 4):
        sky_fig, ground_fig, leaflet_map = make_xst_plots(visibilities, station_name, obstime, subband, rcu_mode, hdf5_filename=f"results/results.h5", extent=[-300, 300, -300, 300], outputpath='/Volumes/home/RFI')
        plt.close('all')
    else:
        logger.warning("Unsupported rcu mode: {}".format(rcu_mode))


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.info("Started collecting from OPCUA")

    while True:
        for station_name in ("RS210", "RS208", "CS103"):
            for subband in (150, 250, 300, 350, 375, 400):
                thread = threading.Thread(target=download_and_image, args=(station_name, subband))
                thread.start()
                time.sleep(30)
