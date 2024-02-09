import ee
import numpy as np
import pandas as pd
from datetime import datetime
#from datetime import timedelta
from dateutil.relativedelta import relativedelta
import gc #garbage collector

import re #expressão regular
from scipy.signal import savgol_filter #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

from sklearn import preprocessing

import specIndex
#import fireAnalysis
import maskingFunctions
#---------------------------------------------


#==============================================
def get_geometry_from_corners(coordenadas):
    
    x1,y1,x2,y2 = coordenadas.split(",")
    geometria = ee.Geometry.Polygon( \
                                    [[[float(x1),float(y2)], \
                                      [float(x2),float(y2)], \
                                      [float(x2),float(y1)], \
                                      [float(x1),float(y1)], \
                                      [float(x1),float(y2)]]])

    return geometria


#==============================================
def ext_lat_lon_pixel(image, geometria, bandas, escala, coordSystem):
    image = image.addBands(ee.Image.pixelLonLat())
    dictImg = image.select(['longitude', 'latitude']+bandas) \
                   .reduceRegion(reducer=ee.Reducer.toList(),\
                                 geometry=geometria,scale=escala,\
                                 bestEffort=True,crs=coordSystem)
    bandas_valores = []
    for banda in bandas:
        bandas_valores.append(np.array(ee.List(dictImg.get(banda)).getInfo()).astype(float))

    return np.array(ee.List(dictImg.get('latitude')).getInfo()).astype(float), np.array(ee.List(dictImg.get('longitude')).getInfo()).astype(float), bandas_valores


#==============================================
#...
def repack_collection_mod09_aq1(startData,endData,geometry,scale,coordSystem,resamplingMethod):
    
    mod09A1 = 'MODIS/006/MOD09A1'
    mod09Q1 = 'MODIS/006/MOD09Q1'
    
    collectionA1 = ee.ImageCollection(mod09A1) \
                     .filterBounds(geometry) \
                     .filterDate(startData,endData)
                    #.filterMetadata('CLOUD_COVER','less_than', cloudCoverMaxPercentage)

    collectionQ1 = ee.ImageCollection(mod09Q1) \
                     .filterBounds(geometry) \
                     .filterDate(startData,endData)
                    #.filterMetadata('CLOUD_COVER','less_than', cloudCoverMaxPercentage)

    #Obtendo as infos das imagens da collectionA1
    dictA1 = {}
    imageListA1 = collectionA1.toList(collectionA1.size())
    itensA1 = imageListA1.size().getInfo()
    for ind in range(itensA1):
        print('Backing up A1: '+str(ind)+'/'+str(itensA1))
        dictA1[ind] = ee.Image(imageListA1.get(ind)).getInfo()

    #Obtendo as infos das imagens da collectionA1
    dictQ1 = {}
    imageListQ1 = collectionQ1.toList(collectionQ1.size())
    itensQ1 = imageListA1.size().getInfo()
    for ind in range(imageListQ1.size().getInfo()):
        print('Backing up Q1: '+str(ind)+'/'+str(itensQ1))
        dictQ1[ind] = ee.Image(imageListQ1.get(ind)).getInfo()

    #Iniciar um reempacotamento dos dias/imagens em uma nova coleção da união de A1 e Q1
    system_index_A1 = [dictA1[index]['properties']['system:index'] for index in dictA1.keys()]
    system_index_Q1 = [dictQ1[index]['properties']['system:index'] for index in dictQ1.keys()]

    imgListPack = []
    for keyA1 in dictA1.keys():
        print('Resampling AQ1: '+str(keyA1)+'/'+str(len(dictA1.keys())))
        keyQ1 = system_index_Q1.index( dictA1[keyA1]['properties']['system:index'] )

        #Reamostragem...
        imgA1 = ee.Image(imageListA1.get(keyA1)).resample(resamplingMethod).reproject(crs=coordSystem, scale=scale)
        imgQ1 = ee.Image(imageListQ1.get(keyQ1)).resample(resamplingMethod).reproject(crs=coordSystem, scale=scale)        

        temp = ee.Image().addBands(imgA1).addBands(imgQ1)
        imgListPack.append( temp )
    
    repackCollection = ee.ImageCollection.fromImages( imgListPack )
    return repackCollection


#==============================================
#...
def build_dataframe(collection,listAtts,geometry,scale,coordSystem,defaultDummy):
    #...

    #Obtendo as infos das imagens da collectionA1
    dictBase = {}
    dictRelational = {}
    imageList = collection.toList(collection.size())
    numItens = imageList.size().getInfo()
    for ind in range(numItens):
        print('Getting instant '+str(ind)+'/'+str(numItens))
        #tempImage = ee.Image(imageList.get(ind))
        tempImage = ee.Image(defaultDummy).blend( ee.Image(imageList.get(ind)) )
        lat, lon, values = ext_lat_lon_pixel(tempImage, geometry, listAtts, scale, coordSystem)

        for attS, attI in zip(listAtts,range(len(listAtts))):
            att_inst = attS+'_'+str(ind)
            dictBase[att_inst] = values[attI]  

    dictBase['Latitude'] = lat
    dictBase['Longitude'] = lon
    
    dfBase = pd.DataFrame(dictBase)
    return dfBase




#==============================================
#Para cada linha (pixel - lat/lon) do dataFrame, aplicar o SavGol
def apply_savgol_dataframe(dfGeral,listAtts,geometry,scale,coordSystem,window_length,polyorder,interpMode):  

    #Instanciação do novo dataFrame que vai conter os dados filtrados
    dfFiltered = dfGeral[['Latitude','Longitude']]

    dfHeads = dfGeral.keys()

    #Seleciona as colunas com o nome do atributo
    for att in listAtts:
        selCols = []
        selColsRename = []     
        exp = re.compile(att+'_\d')
        for item in dfHeads:
            out = exp.match(item)
            if out != None:
                selCols.append(item)
                selColsRename.append(item+'_savgol')

        matData = dfGeral[selCols].to_numpy()
        filData = savgol_filter(matData, window_length, polyorder, axis= -1, mode=interpMode)

        #Depositando os valores filtrados no dataFrame de saída
        for ind in range(filData.shape[1]): dfFiltered[ selColsRename[ind] ] = filData[:,ind]
        
    return dfFiltered
#---------------------------------------------------------------



#==============================================
#...
def compute_delta_nbr(dfBase,thresholdNBR):  

    #Instanciação do novo dataFrame que vai conter os dados filtrados
    dfBaseDelta = dfBase #mera cópia (de segurança?)

    dfHeads = dfBaseDelta.keys()

    #Seleciona as colunas com 'nbr' no nome
    selCols = []
    exp = re.compile('nbr_\d_savgol')
    for item in dfHeads:
        out = exp.match(item)
        if out != None: selCols.append(item)


    #Cálculo do deltaNBR (pré - pós fogo)
    for ind in range(1,len(selCols)):
        diff = dfBaseDelta[ selCols[ind-1] ].to_numpy() - dfBaseDelta[ selCols[ind] ].to_numpy()
        dfBaseDelta[ 'dnbr_'+str(ind-1) ] = diff
        dfBaseDelta[ 'thres_dnbr_'+str(ind-1) ] = (diff > thresholdNBR)


    #Cálculo da frequência de eventos baseados no deltaNBR (alocado em 'sum_fire')
    dfHeads = dfBaseDelta.keys()
    selCols = []
    exp = re.compile('thres_dnbr_\d')
    for item in dfHeads:
        out = exp.match(item)
        if out != None: selCols.append(item)
    dfBaseDelta[ 'sum_fire'] = dfBaseDelta[ selCols ].sum(axis=1)

    return dfBaseDelta




#==============================================
#Adotar um esquema recursivo sobre "np" (que retorna a data cada vez mais, até que encontre alguma imagem)
#Contempla apenas indices espectrais (teste inicial...)
def expand_period_collection(startYDM,endData,T,sensorName,geometry,scale,cloudCoverMaxPercentage,coordSystem):
    
    print("Busca de nivel "+str(T))

    info = specIndex.sensor_info('modis')

    __startYDM = startYDM - relativedelta(months=T) #caso não haja imagem entre startYDM~endData, é subtraido T meses de startYDM

    str_startData = str(__startYDM.year)+'-'+str(__startYDM.month)+'-'+str(__startYDM.day)
    str_endData = str(endData.year)+'-'+str(endData.month)+'-'+str(endData.day)
    
    
    pre_collection = ee.ImageCollection(info['completeSensorName']) \
                       .filterBounds(geometry) \
                       .filterDate(str_startData,str_endData) #\
                       #.filterMetadata('CLOUD_COVER','less_than', cloudCoverMaxPercentage)


    collection = maskingFunctions.repack_collection(pre_collection,            #Coleção sob análise
                                                    geometry,                  #Geometria da coleção
                                                    scale,                     #Escala dos dados
                                                    3,                         #Período (meses) usados na geração de uma base de suporte (img. mediana)
                                                    sensorName,                #Nome do sensor (útil a criação das máscaras de nuvem)
                                                    cloudCoverMaxPercentage,   #Percentual de cobertura de nuvem
                                                    0.5)                       #Utilidade da imagem (>50%)

    found = collection.size().getInfo()

    #Chamada recursiva
    #Enquanto não encontra imagens válidas no intervalo (refData-T ~ refData)
    #o valor de T aumenta mais, expandindo o intervalo de busca para (refData-(T+1) ~ refData)
    if found != 0:
        return collection
    else:
        #Chamada recursiva...
        return expand_period_collection(startYDM,endData,T+1,sensorName,geometry,scale,cloudCoverMaxPercentage,coordSystem)



#==============================================
#Contempla apenas indices espectrais (teste inicial...)
def get_df_inst(refData,
                nPast,
                geometry,
                scale,
                sensorName,
                cloudCoverMaxPercentage,
                coordSystem,
                getFocos):

    defaultDummy = 0.0 #-9999.0

    #Get basic information from the adopted sensor
    infoMSpec = specIndex.sensor_info(sensorName['multispec'])
    infoPrec = specIndex.sensor_info(sensorName['precipitation'])
    infoTemp = specIndex.sensor_info(sensorName['temperature'])
    infoHuEv = specIndex.sensor_info(sensorName['humidity'])     #add 19mar21
    infoFire = specIndex.sensor_info(sensorName['fire'])
    

    #Define a data type useful to manipulate time variable
    ymd = datetime.strptime(refData, "%Y-%m-%d")

    d = {} #Defining a dictionary structure to build a dataFrame at end of process
    #dFire = {} #Defining a dictionary structure to build a dataFrame for fire spots analysis

    #Loop over the "nPast" time range (in months)
    for t in range(1,nPast+1):
        
        #No tratamento de valores faltantes (devido a cobertura de nuvens, etc), 
        # talvez seja necessário criar uma imagem "interpoladora"
        # (e.g., a mediana de uma dada faixa de meses que flutura com esses loop)

        #Data final para a coleção do mês (colocar sempre último dia do mês? PENSAR)
        endYDM = ymd - relativedelta(months=t-1)
        endData = str(endYDM.year)+'-'+str(endYDM.month)+'-'+str(endYDM.day)

        #Util no caso dos dados de precipitacao e temperatura
        startYDM = ymd - relativedelta(months=t)
        startData = str(startYDM.year)+'-'+str(startYDM.month)+'-'+str(startYDM.day)

        #Dados multiespectrais-------------------------------
        #monthCollection = expand_period_collection(endData,t,geometry,scale,sensorName,cloudCoverMaxPercentage,coordSystem)
        #O 2o argumento sempre inicia em 1, pois é o salto de tempo que é dado a partir de "endData"
        #monthCollection = expand_period_collection(endData,1,geometry,scale,sensorName['multispec'],cloudCoverMaxPercentage,coordSystem)
        monthCollection = expand_period_collection(endData,1,geometry,scale,sensorName,cloudCoverMaxPercentage,coordSystem)

        #Cálculo de índices espectrais
        monthCollection = specIndex.add_ndvi_collection(monthCollection,sensorName['multispec'])
        monthCollection = specIndex.add_ndwi_collection(monthCollection,sensorName['multispec'])
        monthCollection = specIndex.add_savi_collection(monthCollection,sensorName['multispec'])
        monthCollection = specIndex.add_nbr_collection(monthCollection,sensorName['multispec'])

        #Cálculo da "mediana espectral" no mês
        medianMonth = monthCollection.median()

        #Extração dos dados sobre índices espectrais
        #Utilizando lonMS/latMS em distinção aos demais para fins de teste (e inclusão no dict ao final do processo)
        temp = ee.Image(defaultDummy).blend(ee.Image( medianMonth ))
        latMS, lonMS, indNDVI = ext_lat_lon_pixel(temp, geometry, ['ndvi'], scale, coordSystem)
        latMS, lonMS, indNDWI = ext_lat_lon_pixel(temp, geometry, ['ndwi'], scale, coordSystem)
        latMS, lonMS, indSAVI = ext_lat_lon_pixel(temp, geometry, ['savi'], scale, coordSystem)  #Add 19mar21
        latMS, lonMS, indNBR  = ext_lat_lon_pixel(temp, geometry, ['nbr'], scale, coordSystem)   #Add 19mar21


        #Dados de precipitação-------------------------------
        monthPrecipitation = ee.ImageCollection(infoPrec['completeSensorName']) \
                               .filterBounds(geometry) \
                               .filterDate(startData,endData)

        #Cálculo da precipitação acumulada no mês
        cumMonthPrec = monthPrecipitation.sum().rename('precipitation') #<<<funciona pq tem uma banda só!!
        
        #Fazer a adição da precipitação do mês
        temp = temp.addBands( ee.Image(cumMonthPrec) )

        #Extração dos dados sobre preciptação acumulada
        lat, lon, indPrec = ext_lat_lon_pixel(temp, geometry, ['precipitation'], scale, coordSystem)



        #Dados de temperatura----------------------------------
        monthTemperature = ee.ImageCollection(infoTemp['completeSensorName']) \
                             .filterBounds(geometry) \
                             .filterDate(startData,endData)

        #Cálculo da temperatura média no mês
        #avgMonthTemp = monthTemperature.mean().rename('temperature')
        #"Select" apenas na banda de interesse... ('temperature_2m')
        avgMonthTemp = monthTemperature.mean().select(infoTemp['bands']['temperature_2m']).rename('temperature')
        
        #Fazer a adição da temperatura do mês
        temp = temp.addBands( ee.Image(avgMonthTemp) )

        #Extração dos dados sobre preciptação acumulada
        lat, lon, indTemp = ext_lat_lon_pixel(temp, geometry, ['temperature'], scale, coordSystem)


        
        #Dados de evapotranspiração e umidade----------------------
        monthHumidtyEvapo = ee.ImageCollection(infoHuEv['completeSensorName']) \
                              .filterBounds(geometry) \
                              .filterDate(startData,endData)

        #Cálculo da evapotranspiração e humidade média no mês
        avgMonthHuEv = monthHumidtyEvapo.mean().select( \
                                                         [ infoHuEv['bands']['specificHumidity'] ,   \
                                                           infoHuEv['bands']['Evapotranspiration'] ] \
                                                      ).rename(['humidity','evapotransp'])

        #Fazer a adição da umidade e evapotranspiração do mês
        temp = temp.addBands( ee.Image(avgMonthHuEv) )

        #Extração dos dados sobre umidade e evapotranspiração média do mês
        lat, lon, indHuEv = ext_lat_lon_pixel(temp, geometry, ['humidity','evapotransp'], scale, coordSystem)




        #>>> SE A KEYWORD ESTIVER ATIVADA!
        if getFocos:
            #Dados do produto FIRMS---------------------------------
            monthFire = ee.ImageCollection(infoFire['completeSensorName']) \
                          .filterBounds(geometry) \
                          .filterDate(startData,endData)

            #lon, lat, indNDVI = ext_lat_lon_pixel(temp, geometry, ['ndvi'], scale, coordSystem)
            monthFire = specIndex.add_firespots_collection(monthFire,sensorName['fire'])

            countMonthFires = monthFire.sum().select('occurrence').rename('contOccurrence')
            #infoTemp['bands']['temperature_2m']

            temp = temp.addBands( ee.Image(countMonthFires) )
            #Extração dos dados sobre os focos de incêndio
            lat, lon, indFocos = ext_lat_lon_pixel(temp, geometry, ['contOccurrence'], scale, coordSystem)
        #-----------------------------------------------


    
        d['NDVI'+str(t)] = indNDVI[0]    #<<< [0] for compatibility (this is a numpyArray)
        d['NDWI'+str(t)] = indNDWI[0]
        
        d['SAVI'+str(t)] = indSAVI[0]   #Add 19mar21
        d['NBR'+str(t)] = indNBR[0]     #Add 19mar21
        
        d['Prec'+str(t)] = indPrec[0]
        d['Temp'+str(t)] = indTemp[0]    #<<< estou achando estranho os valores, que deveriam estar em kelvin
        
        d['Humi'+str(t)] = indHuEv[0] * (10**2)   #Add 19mar21 (uso de fator mutiplicativo)
        d['Evap'+str(t)] = indHuEv[1] * (10**5)   #Add 19mar21

        #Armazenar a informação sobre a ocorrência de focos
        if getFocos: d['Foco'+str(t)] = indFocos[0]

        print('>>>'+str(t))



        gc.collect() #Coletor de lixo :: Precisa desalocar a var. "temp"?


    #Inclusão da lat/lon no dict
    d['Latitude'] = latMS
    d['Longitude'] = lonMS

    #Construct and fill in the resulting dataFrame
    tab = pd.DataFrame()
    tab = tab.from_dict(d)   #<<<Algumas variáveis (prec e temp) não preenchem toda a geometria... isso pode causar probelma aqui!


    if getFocos:
        #tabRev = fireAnalysis.fire_spots_df_v2(tab,refData)#,gmmComponents)
        tabRev = fireAnalysis.fire_spots_df_v3(tab,refData)
        return tabRev
    else:
        return tab

