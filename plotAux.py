import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

#Melhorar essa fução e/ou fazer outras... é apenas um esboço inicial!

def plot_check_map(dataFrame,path_out,colName):

    FS = (10,10) #Tamanho da figura a ser gerada
    fig = plt.figure(constrained_layout=True,figsize=FS)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1 = dataFrame.plot.scatter(x='Longitude', y='Latitude', c=colName, colormap='viridis')
    ax1.set_aspect('equal')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')#, format='pdf')
    #plt.show()


def plot_check_map_threshold(dataFrame,path_out,colName,thresold):
    FS = (10,10) #Tamanho da figura a ser gerada
    fig = plt.figure(constrained_layout=True,figsize=FS)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])

    #dataFrame2 = dataFrame.loc[ : , colName ]
    dataFrame2 = dataFrame.loc[ dataFrame.loc[:,colName] > thresold , : ]

    ax1 = dataFrame2.plot.scatter(x='Longitude', y='Latitude', c=colName, colormap='viridis')
    ax1.set_aspect('equal')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')#, format='pdf')
    #plt.show()


def scatter_attributes(dataFrame,path_out,listAtts):
    FS = (10,10) #Tamanho da figura a ser gerada
    fig = plt.figure(constrained_layout=True,figsize=FS)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1 = dataFrame.plot.scatter(x=listAtts[0], y=listAtts[1], c=listAtts[0], colormap='viridis')
    ax1.set_aspect('equal')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')#, format='pdf')



def boxplot_attributes(dataFrame,path_out,listAtts):
    FS = (10,10) #Tamanho da figura a ser gerada
    fig = plt.figure(constrained_layout=True,figsize=FS)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    #ax1 = dataFrame.plot.scatter(x=listAtts[0], y=listAtts[1], c=listAtts[0], colormap='viridis')
    ax1 = dataFrame.boxplot(column=listAtts)
    #ax1.set_aspect('equal')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')#, format='pdf')



def histogram_attributes(dataFrame,path_out,listAtts,bins):
    FS = (10,10) #Tamanho da figura a ser gerada
    fig = plt.figure(constrained_layout=True,figsize=FS)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    
    ax1 = dataFrame.hist(column=listAtts, bins=bins)
    
    plt.savefig(path_out, dpi=300, bbox_inches='tight')



def print_list_info_collection(collection):

    dim = collection.size().getInfo()
    imageList = collection.toList(collection.size())
    for i in range(dim):
        imageInfo = ee.Image(imageList.get(i)).getInfo()
        print(str(i)+' --> '+imageInfo['properties']['SENSING_TIME'].split('T')[0])