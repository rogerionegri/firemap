import numpy as np
from osgeo import gdal
from osgeo import osr

#===============================================
def save_tiff_from_df(df,bands,dummy,path_out,coordSystem):
        
    #Registro das Lats/Lons
    lat = df['Latitude'].values
    lon = df['Longitude'].values
    
    #Determinação de algumas variáveis de interesse
    ulat = np.unique(lat)
    ulon = np.unique(lon)
    ncols = len(ulon)
    nrows = len(ulat)
    nbands = len(bands)       #Lista com o nome das bandas que deseja utilizar na montagem da imagem
    
    #Variação base em linha/coluna
    ys = ulat[11]-ulat[10]
    xs = ulon[11]-ulon[10]    
    
    #Determinação de um array que compreenderá a imagem
    arr = np.zeros([nbands, nrows, ncols], np.float32)
    refLat = np.max(ulat)
    refLon = np.min(ulon)
    for j in range(len(df)):
        posLin = np.int64( np.round( (refLat - lat[j])/ys ) )
        posCol = np.int64( np.round( (lon[j] - refLon)/xs ) )
        for b in range(nbands):
            arr[b,posLin,posCol] = df.loc[df.index[j],bands[b]]
            
    transform = (np.min(ulon),xs,0,np.max(ulat),0,-ys)
    target = osr.SpatialReference()
    
    #Determinação do sistema de coordenadas
    #target.ImportFromEPSG(4326)
    #target.ImportFromEPSG( np.int64(coordSystem.split(':')[1]) )
    target.ImportFromEPSG( int(coordSystem.split(':')[1]) )
    
    #Determinando o driver e outras especificações
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(path_out,ncols,nrows,nbands,gdal.GDT_Float32)
    outDs.SetGeoTransform(transform)
    outDs.SetProjection(target.ExportToWkt())

    #Montatem...
    ind = 1
    for b in range(nbands):
        bandArr = np.copy(arr[b,:,:])
        outBand = outDs.GetRasterBand(ind)
        outBand.WriteArray(bandArr)
        outBand.FlushCache()
        outBand.SetNoDataValue(dummy)
        ind += 1

    outDs = None
    del outDs, outBand

    return True