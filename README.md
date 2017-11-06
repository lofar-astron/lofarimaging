# LOFAR LBA imaging tutorial
Basic LOFAR imaging with an LBA station, based on the tutorial provided by Michiel Brentjens (github.com/brentjens). This notebook is compatible with Python 2.7.

### To start the notebook
`>> jupyter notebook`

### To run the entire notebook
`Kernel > Restart & Run All`

### Notes
This ships with an example dataset (20170720_095816_xst.dat), obtained using station DE603LBA, specifically collecting data in subband 297 (58 MHz). It should run on any other dataset, provided the station name and subband are changed accordingly. By default, it visualises timestep #5 of the 30 timesteps of integration, but this can also be modified to any other time slice. 

### Release: Version 3
Notebook has been updated to include station calibration for LBA (HBA to come later). This requires the "caltables" folder to be in the same directory as the notebook, but that can be changed as long as the location is changed also in the function. The calibration tables themselves are not included in this repository due to their size, but you can access them here (~65 MB download): http://astron.nl/~moss/caltables.zip
