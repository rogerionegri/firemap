#Imports-------------------------------------
import ee
from osgeo import gdal
from osgeo import osr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from dateutil import relativedelta
from datetime import datetime

#Local modules-----------------------------
import specIndex
import dataManage
import plotAux
import saveModule

ee.Initialize() #Init. API Earth Engine
#---------------------------------------------

#General parameters---------------------------
cloudCoverMaxPercentage = 10.0 
scale = 250 #pixel size
resamplingMethod = 'bilinear'
coordSystem = "epsg:4326"

#Savitzky–Golay filtering parameters
window_length = 11  #---> positive integer
polyorder = 5
interpMode = 'interp'

#COnsidered attributes
listAtts = ['ndvi','nbr']  #'ndvi','ndwi','nbr', 'savi'

#Threshold for deltaNBR to distinguish 'burn/scars'
thresholdNBR = 0.1

#------------------------------------
startData = "2020-05-01"  #Database start data
endData =   "2021-05-01"  #...end data

#Area of interest
coordenadas = "-59.200759,-15.441264,-57.0395,-22.198933"

#Output path
path_res = './'
#---------------------------------------------

defaultDummy = -9999.0 #Dummy value to represent 'NoData'


geometry = dataManage.get_geometry_from_corners(coordenadas)
infoMSpec = specIndex.sensor_info('modisPackA1Q1')
collectionModisAQ1 = dataManage.repack_collection_mod09_aq1(startData,
                                                            endData,
                                                            geometry,
                                                            scale,
                                                            coordSystem,
                                                            resamplingMethod)

#Add spectral indices--------------
collectionModisAQ1 = specIndex.add_ndvi_collection(collectionModisAQ1,'modisPackA1Q1')
collectionModisAQ1 = specIndex.add_ndwi_collection(collectionModisAQ1,'modisPackA1Q1')
collectionModisAQ1 = specIndex.add_nbr_collection(collectionModisAQ1,'modisPackA1Q1')


#Dataframe construction
dfGeral = dataManage.build_dataframe(collectionModisAQ1,listAtts,geometry,scale,coordSystem,defaultDummy)


#----------------------------------------------------
#Apply the SG filtering
dfGeral = dataManage.apply_savgol_dataframe(dfGeral,listAtts,geometry,scale,coordSystem,window_length,polyorder,interpMode)

#Computing the deltaNBR index
dfGeral = dataManage.compute_delta_nbr(dfGeral,thresholdNBR)

#Plotting the output
plotAux.plot_check_map(dfGeral,path_res+'map0.png','sum_fire')


#Saving the output
path_save = path_res + 'outputMap.tif'
saveModule.save_tiff_from_df(dfGeral,['sum_fire'],defaultDummy,path_save,coordSystem)

print('End of process...')

