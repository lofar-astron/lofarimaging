#!/usr/bin/env python

import time
from lofarimaging.opc_interface import download_xst
from lofarimaging import make_xst_plots
import threading
import sys
import matplotlib.pyplot as plt

integration_time_s = 5

def download_and_image(station_name, subband):
    port = 50000 + int(station_name[2:])
    obstime, visibilities, rcu_mode = download_xst(subband, integration_time_s, port=port)
    if rcu_mode in (1, 2, 3, 4):
        sky_fig, ground_fig, leaflet_map = make_xst_plots(visibilities, station_name, obstime, subband, rcu_mode, hdf5_filename=f"results/results.h5", extent=[-300, 300, -300, 300], outputpath='/Volumes/home/RFI')
        plt.close('all')
    else:
        print("Unsupported rcu mode:", rcu_mode)


if __name__ == "__main__":
    while True:
        for station_name in ("RS210", "RS208", "CS103"):
            for subband in (150, 250, 350, 375, 400, 425):
                thread = threading.Thread(target=download_and_image, args=(station_name, subband))
                thread.start()
                time.sleep(30)
