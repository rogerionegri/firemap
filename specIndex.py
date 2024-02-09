

#Get infos from selected sensor--------------------------------
def sensor_info(sensorName):

    #completar ao passo que for necessário...

    if sensorName == 'landsat':
        #defaultSensorName = "USGS Landsat 8 Surface Reflectance Tier 1 - LANDSAT/LC08/C01/T1_SR"
        defaultSensorName = "LANDSAT/LC08/C01/T1_SR" #"LANDSAT/LC08/C01/T1_TOA"
        defaultScale = 30
        #blue = 'B2'
        #red = 'B4'
        #green = 'B3'
        #swir = 'B6'
        #nir = 'B5'
        #c_nir = 865
        #c_red = 654.5
        #c_swir = 1608.5
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'blue': 'B2', 'red': 'B4', 'green': 'B3', 'swir1': 'B6', 'swir2': 'B7', 'nir': 'B5'}, \
                'centerBands': {'nir': 865.0, 'red': 654.5, 'swir': 1608.5},\
                'quality': 'pixel_qa',\
                'BoI': ['B2','B3','B4','B5','B6','B7']     }

    if sensorName == 'sentinel':
        #defaultSensorName = "Sentinel-2 MSI: MultiSpectral Instrument, Level-2A - COPERNICUS/S2_SR"
        defaultSensorName = "COPERNICUS/S2_SR"
        defaultScale = 30
        #blue = 'B2'
        #red = 'B4'
        #green = 'B3'
        #swir = 'B11'
        #nir = 'B8'
        #c_nir = 834.05
        #c_red = 664.75
        #c_swir = 1612.05
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'blue': 'B2', 'red': 'B4', 'green': 'B3', 'swir': 'B11', 'nir': 'B8'}, \
                'centerBands': {'nir': 834.05, 'red': 664.75, 'swir': 1612.05}, \
                'BoI': ['B2','B3','B4','B8','B11']     }

    if sensorName == 'modis':
        #defaultSensorName = "MOD09GA.006 Terra Surface Reflectance Daily L2G Global 1km and 500m - MODIS/006/MOD09GA"
        defaultSensorName = "MODIS/006/MOD09GA"
        defaultScale = 500
        blue = 'sur_refl_b03'
        red = 'sur_refl_b01'
        green = 'sur_refl_b04'
        swir = 'sur_refl_b06'
        nir = 'sur_refl_b02'
        c_nir = 858.5
        c_red = 645
        c_swir = 1640
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'blue': 'sur_refl_b03', 'red': 'sur_refl_b01', 'green': 'sur_refl_b04', 'swir': 'sur_refl_b06', 'nir': 'sur_refl_b02'}, \
                'centerBands': {'nir': 858.5, 'red': 645.0, 'swir': 1640.0},\
                'quality': 'StateQA', \
                'BoI': ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06']       }


    if sensorName == 'modisPackA1Q1':
        #...Compilado de dois produtos...
        #['constant', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'QA', 'SolarZenith', 'ViewZenith', 'RelativeAzimuth', 'StateQA', 'DayOfYear', 'sur_refl_b01_1', 'sur_refl_b02_1', 'State', 'QA_1']
        defaultSensorName = ['MODIS/006/MOD09A1','MODIS/006/MOD09Q1']
        defaultScale = 250
        blue = 'sur_refl_b03'
        red = 'sur_refl_b01'
        green = 'sur_refl_b04'
        swir2 = 'sur_refl_b07'
        nir = 'sur_refl_b02'
        #c_nir = 858.5
        #c_red = 645
        #c_swir = 1640
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'blue': blue, 'green': green, 'red': red, 'nir': nir, 'swir2': swir2}, \
                #'centerBands': {'nir': 858.5, 'red': 645.0, 'swir': 1640.0},\
                'quality': 'StateQA', \
                #'BoI': ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06']       }
                'BoI': [blue,green,red,nir,swir2]  }





    if sensorName == 'chirps':
        #defaultSensorName = "CHIRPS Daily: Climate Hazards Group InfraRed Precipitation with Station Data (version 2.0 final)"
        defaultSensorName = "UCSB-CHG/CHIRPS/DAILY"
        defaultScale = 0.05 #arc degrees
        data = 'precipitation'
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'precipitation': 'precipitation'}  }
    

    if sensorName == 'era5':
        #defaultSensorName = "ERA5-Land monthly averaged - ECMWF climate reanalysis"
        #defaultSensorName = "ECMWF/ERA5/MONTHLY"
        #defaultSensorName = "ECMWF/ERA5/DAILY"
        defaultSensorName = "ECMWF/ERA5_LAND/MONTHLY"
        defaultScale = 0.01 #arc degrees
        data = 'temperature'
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'temperature_2m': 'temperature_2m'}  } #other variables were disregarded

    if sensorName == 'firms':
        #defaultSensorName = "FIRMS: Fire Information for Resource Management System"
        defaultSensorName = "FIRMS"
        defaultScale = 1000 #meters
        data = 'fire' #'temperature'   #<<<<<<<<<???
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'brightModis21': 'T21', 'confidence': 'confidence', 'lineNumber': 'line_number'},\
                'thresholdK': 350, 'thresholdConf': 90 }


    #Adicionado em 19mar21
    if sensorName == 'fldas':
        #defaultSensorName = FLDAS: Famine Early Warning Systems Network (FEWS NET) Land Data Assimilation System
        defaultSensorName = "NASA/FLDAS/NOAH01/C/GL/M/V001"
        defaultScale = 0.1 #arc degrees
        data = 'humidity'
        return {'completeSensorName': defaultSensorName, 'defaultScale': defaultScale, \
                'bands': {'specificHumidity': 'Qair_f_tavg', 'Evapotranspiration': 'Evap_tavg'} }





#Add NDVI to each image in the collection
def add_ndvi_collection(collection,sensorName):
    info = sensor_info(sensorName)
    
    #Local function
    def ndvi(image):
        dictBands = {'nir': image.select(info['bands']['nir']), 'red': image.select(info['bands']['red']) }
        ndvi = image.expression('(nir - red) / (nir + red)',dictBands).rename('ndvi')
        return image.addBands(ndvi)

    collection = collection.map(ndvi)
    return collection


#Add NDWI to each image in the collection
def add_ndwi_collection(collection,sensorName):
    info = sensor_info(sensorName)
    
    #Local function
    def ndwi(image):
        dictBands = {'nir': image.select(info['bands']['nir']), 'green': image.select(info['bands']['green']) }
        ndwi = image.expression('(nir - green) / (nir + green)',dictBands).rename('ndwi')
        return image.addBands(ndwi)

    collection = collection.map(ndwi)
    return collection


#Add SAVI to each image in the collection
#   SAVI is calculated as a ratio between the R and NIR values with a soil brightness correction factor (L) 
#   ...defined as 0.5 to accommodate most land cover types.
#   https://www.usgs.gov/core-science-systems/nli/landsat/landsat-soil-adjusted-vegetation-index
def add_savi_collection(collection,sensorName):
    info = sensor_info(sensorName)
    
    #if sensorName == 'landsat': factor = 1.5
    #if sensorName == 'modis': factor = 1.5    #???????

    #Local function
    def savi(image):
        dictBands = {'nir': image.select(info['bands']['nir']), 'red': image.select(info['bands']['red']) }
        savi = image.expression('((nir - red) / (nir + red + 0.5))*1.5',dictBands).rename('savi')
        return image.addBands(savi)

    collection = collection.map(savi)
    return collection


#Add MBR to each image in the collection
#   NBR is used to identify burned areas and provide a measure of burn severity. 
#   It is calculated as a ratio between the NIR and SWIR values in traditional fashion.
#   https://www.usgs.gov/core-science-systems/nli/landsat/landsat-normalized-burn-ratio
def add_nbr_collection(collection,sensorName):
    info = sensor_info(sensorName)
    
    #Local function
    def nbr(image):
        dictBands = {'nir': image.select(info['bands']['nir']), 'swir2': image.select(info['bands']['swir2']) }
        nbr = image.expression('(nir - swir2) / (nir + swir2)',dictBands).rename('nbr')
        return image.addBands(nbr)

    collection = collection.map(nbr)
    return collection



#Add "Focos" to each image in the collection
def add_firespots_collection(collection,sensorName):
    info = sensor_info(sensorName)
    
    #Local function
    def spots_firms(image):
        dictBands = {'brightModis21': image.select( info['bands']['brightModis21'] ), \
                     'confidence':    image.select( info['bands']['confidence'] ), \
                     'thresholdK':    info['thresholdK'], \
                     'thresholdConf': info['thresholdConf'] }
        
        stringOper = '(brightModis21 >= thresholdK) and (confidence >= thresholdConf) ? 1 : 0'

        occurrence = image.expression(stringOper,dictBands).rename('occurrence')        
        
        return image.addBands(occurrence)

    collection = collection.map(spots_firms)
    return collection

