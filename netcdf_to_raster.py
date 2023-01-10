import xarray as xr 
import rioxarray as rio 
import rasterio
import rasterio.plot
import pyproj

"""
# import rioxarray and shapley
import rioxarray as riox
from shapely.geometry import Polygon
 
# Read raster using rioxarray
raster = riox.open_rasterio('cal_census.tiff')
 
# Shapely Polygon  to clip raster
geom = Polygon([[-13315253,3920415], [-13315821.7,4169010.0], [-13019053.84,4168177.65], [-13020302.1595,3921355.7391]])
 
# Use shapely polygon in clip method of rioxarray object to clip raster
clipped_raster = raster.rio.clip([geom])
 
# Save clipped raster
clipped_raster.rio.to_raster('clipped.tiff')
"""

file = xr.open_dataset("C:/Users/erpasten/Documents/UEF/Hydropower/data/climate/whalley/tasmax_hadukgrid_uk_1km_day.nc")

# Read the file in order to get an idea of the contents, naming and arrangement of the data
# This is important because the rest of the code calls the variables based on the defined naming
file

file_tasmax = file['tasmax']
file_tasmax = file_tasmax.rio.set_spatial_dims(x_dim='projection_x_coordinate', y_dim='projection_y_coordinate')
file_tasmax.rio.crs
file_tasmax.rio.write_crs("epsg:27700", inplace=True)

file_tasmax.rio.to_raster('C:/Users/erpasten/Documents/UEF/Hydropower/figures/Test_raster.tiff')

data_name = ('C:/Users/erpasten/Documents/UEF/Hydropower/figures/Test_raster.tiff')
tiff = rasterio.open(data_name)
rasterio.plot.show(tiff, title = "Max Temperature")

print('bounds:',tiff.bounds)# indicates the spatial bounding box
print('number of bands:',tiff.count)# number of bands
print('number of columns:',tiff.width)# number of columns of the raster dataset
print('number of rows:',tiff.height)# number of rows of the raster
print('coordinate reference system:',tiff.crs)# crs


