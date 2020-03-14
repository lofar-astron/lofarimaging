"""Functions for working with LOFAR single station data"""

import os

import folium
import numpy as np
from matplotlib.pyplot import imread
from owslib.wmts import WebMapTileService
import mercantile

__all__ = ["get_map", "make_leaflet_map"]

__version__ = "1.5.0"


def get_map(lon_min, lon_max, lat_min, lat_max, zoom=19):
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
    upperleft_tile = mercantile.tile(lon_min, lat_max, zoom)
    xmin, ymin = upperleft_tile.x, upperleft_tile.y
    lowerright_tile = mercantile.tile(lon_max, lat_min, zoom)
    xmax, ymax = lowerright_tile.x, lowerright_tile.y

    total_image = np.zeros([256 * (ymax - ymin + 1), 256 * (xmax - xmin + 1), 3], dtype='uint8')

    os.makedirs("tilecache", exist_ok=True)

    tile_min = mercantile.tile(lon_min, lat_min, zoom)
    tile_max = mercantile.tile(lon_max, lat_max, zoom)

    wmts = WebMapTileService("http://server.arcgisonline.com/arcgis/rest/" +
                             "services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml")

    for x in range(tile_min.x, tile_max.x + 1):
        for y in range(tile_max.y, tile_min.y + 1):
            tilename = os.path.join("tilecache", f"World_Imagery_{zoom}_{x}_{y}.jpg")
            if not os.path.isfile(tilename):
                tile = wmts.gettile(layer="World_Imagery", tilematrix=str(zoom), row=y, column=x)
                out = open(tilename, "wb")
                out.write(tile.read())
                out.close()
            tile_image = imread(tilename)
            total_image[(y - ymin) * 256: (y - ymin + 1) * 256,
                        (x - xmin) * 256: (x - xmin + 1) * 256] = tile_image

    total_llmin = {'lon': mercantile.bounds(xmin, ymax, zoom).west, 'lat': mercantile.bounds(xmin, ymax, zoom).south}
    total_llmax = {'lon': mercantile.bounds(xmax, ymin, zoom).east, 'lat': mercantile.bounds(xmax, ymin, zoom).north}

    pix_xmin = int(round(np.interp(lon_min, [total_llmin['lon'], total_llmax['lon']], [0, total_image.shape[1]])))
    pix_ymin = int(round(np.interp(lat_min, [total_llmin['lat'], total_llmax['lat']], [0, total_image.shape[0]])))
    pix_xmax = int(round(np.interp(lon_max, [total_llmin['lon'], total_llmax['lon']], [0, total_image.shape[1]])))
    pix_ymax = int(round(np.interp(lat_max, [total_llmin['lat'], total_llmax['lat']], [0, total_image.shape[0]])))

    return total_image[total_image.shape[0] - pix_ymax: total_image.shape[0] - pix_ymin, pix_xmin: pix_xmax]


def make_leaflet_map(overlay_array: np.array, lon_center: float, lat_center: float, lon_min: float, lat_min: float,
                     lon_max: float, lat_max: float, opacity=0.7):
    """
    Show an image in a leaflet map, using Folium

    Args:
        overlay_array: array with values to be used as overlay
        lon_center: Longitude of center, in degrees
        lat_center: Latitude of center, in degrees
        lon_min: Longitude of lower left corner, in degrees
        lat_min: Latitude of lower left corner, in degrees
        lon_max: Longitude of top right corner, in degrees
        lat_max: Latitude of top right corner, in degrees
        opacity: Opacity of overlay

    Returns:
        An interactive map object that will render nicely in a Jupyter notebook.
    """
    m = folium.Map(location=[lat_center, lon_center], zoom_start=19,
                   tiles='http://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/MapServer/' +
                         'tile/{z}/{y}/{x}',
                   attr='ESRI')
    folium.TileLayer(tiles="OpenStreetMap").add_to(m)

    folium.raster_layers.ImageOverlay(
        name='Near field image',
        image=overlay_array,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    folium.LayerControl().add_to(m)

    return m
