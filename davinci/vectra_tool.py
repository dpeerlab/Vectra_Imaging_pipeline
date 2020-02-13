 ## Libraries

import numpy as np
import PIL
from PIL import Image,ImageSequence
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import scipy.spatial 
import imageio
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import networkx as nx
from collections import defaultdict
import scipy.ndimage as ndimage
import os
from skimage import filters, exposure
import tifffile
import math
from scipy.stats import wilcoxon, ttest_rel
import seaborn as sns
import scipy
from skimage.measure import label, regionprops
import scipy.stats as st






## Read and Writes

def read_tiff(path):
    im = Image.open(path)
    tiff_arr = []
    number_of_channels=7
    width, height = im.width,im.height
    for i, page in enumerate(ImageSequence.Iterator(im)):
        raw_height,raw_width = max([ e[1][2] for e in page.tile]),max([ e[1][3] for e in page.tile])
        im.size = raw_height,raw_width
        tiff_arr.append(np.array(page)[:height,:width])
        if i==6:
            break
    return np.array(tiff_arr)
    
    
## Mask analysis

def mask_to_rle_nonoverlap(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def rle2mask(file):
    with open(file,'r') as OpenRLE:
        shape = [int(x) for x in OpenRLE.readline().split(',')]
        mask=[]
        #print(shape)
        #shape=[1004,1340]
        
        for i,line in enumerate(OpenRLE):
            #print(shape)
            #print((' '.join(line.split(',')[1].split(' ')[1:])))
            mask.append(rle_decode(' '.join(line.split(',')[1].split(' ')[1:]),shape))
            #plt.imshow()
            #break
    return mask



def simplex2edge(simplex):
    """Summary or Description of the Function
    Parameters:
    simplex : simplices from the Delaunay
    Returns:
    edgelist

   """
    #edgelist = []
    edgelist= [[(x[0],x[1]),(x[1],x[0]),(x[0],x[2]),(x[2],x[0]),(x[1],x[2]),(x[2],x[1])] for x in simplex]
    edgelist = list(set([y for x in edgelist for y in x]))
    edgelist = [x for x in edgelist if (x[0] > x[1])]
    return edgelist


def cell_type(df,cell_type_list,cell_type_channel):
    cell_dic = {key:i for i,key in enumerate([''.join((map(str,x))) for x in (cell_type_list)])}
    return np.array([cell_dic[''.join((map(str,x)))] for x in (df[cell_type_channel]).values])




def plot_chord(designedfeature_200,sqrt=False,neighborname=[],dpi=100,ax=None):
    if ax==None:
        plt.figure(figsize=(6,6),dpi=dpi)
        ax=plt.axis([0,0,1,1])
    if len(neighborname)==0:
        neighborname = np.arange(len(designedfeature_200.shape[0]))
    LW = 0.3
    #neighborname = ['Tumor', 'CD8 T cell', 'Treg', 'GITR', 'IDO','Others']
    #nodePos = chordDiagram(flux, ax, colors=[hex2rgb(x) for x in ['#666666', '#66ff66', '#ff6666', '#6666ff']])
    if sqrt:
        nodePos = chordDiagram((designedfeature_200+1), ax)
    else:
        nodePos = chordDiagram(np.sqrt(designedfeature_200+1), ax)
    ax.axis('off')
    prop = dict(fontsize=16*0.8, ha='center', va='center')
    nodes = neighborname
    for i in range(len(neighborname)):
        ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
    # plt.savefig("example.png", dpi=600,
    #         transparent=True,
    #         bbox_inches='tight', pad_inches=0.02)




def build_network(df_all,img_id):
    img1 = df_all[df_all['id_image']==img_id].reset_index()
    cell_points = np.array([np.array(eval(x)) for x in img1['centroid']])
    tri = scipy.spatial.Delaunay(cell_points) 
    edgelist = simplex2edge(tri.simplices)
    G=nx.Graph()
    G.add_nodes_from(np.arange(img1.shape[0]))
    G.add_edges_from(edgelist)
    posdic = dict( zip(np.arange(img1.shape[0]),np.fliplr(cell_points)))
    #flipped_pos = {node: (x,y) for (node, (x,y)) in posdic.items()}
    fixed_nodes = np.arange(img1.shape[0])
    pos = nx.spring_layout(G,pos=posdic, fixed =fixed_nodes)
    G.remove_edges_from(tri.convex_hull)
    G.remove_nodes_from(list(set([x for y in tri.convex_hull for x in y])))
    return G



def build_marker_neighborhood(df_all,img_id,plot=False,plot_image=False):
    patient1_images = sorted(glob.glob('/Users/xiey/GoogleDriveCornell/Data/Lung/Vectra_32RGB/*/*.png'))
    img1 = df_all[df_all['id_image']==img_id].reset_index()
    img1_open = imageio.imread(patient1_images[img_id])
    cell_points = np.array([np.array(eval(x)) for x in img1['centroid']])
    tri = scipy.spatial.Delaunay(cell_points) 
    # Delaunay triangulation
    edgelist = simplex2edge(tri.simplices)
    if plot:
        fig=plt.figure(figsize=(15,15),dpi=200,frameon='off')
        #ax1 = fig.add_subplot(111)
        if plot_image:
            fig=plt.imshow(img1_open,zorder=1)
        else:
            fig=plt.imshow(np.zeros(img1_open.shape),zorder=1)
        plt.triplot(cell_points[:,1], cell_points[:,0], tri.simplices,c='gray',linewidth=0.5,zorder=2)
        plt.scatter(cell_points[:,1], cell_points[:,0],c='white',s=3,zorder=3)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        #plt.savefig('../slides/2019-March/network_point.png', bbox_inches='tight', pad_inches=0)
    G=nx.Graph()
    G.add_nodes_from(np.arange(img1.shape[0]))
    G.add_edges_from(edgelist)
    posdic = dict( zip(np.arange(img1.shape[0]),np.fliplr(cell_points)))
    flipped_pos = {node: (x,y) for (node, (x,y)) in posdic.items()}
    fixed_nodes = np.arange(img1.shape[0])
    pos = nx.spring_layout(G,pos=flipped_pos, fixed =fixed_nodes)
    G.remove_edges_from(tri.convex_hull)
    G.remove_nodes_from(list(set([x for y in tri.convex_hull for x in y])))
    neighborname = ['CK', 'CD8', 'FOXP3', 'GITR', 'IDO','Others']
    neighborhoodslist = ['CK/red', 'CD8/yellow', 'FOXP3/pink', 'GITR/green','IDO/white']                   
    designedfeature = np.zeros((len(neighborhoodslist)+1,len(neighborhoodslist)+1))
    for edge in list(G.edges):
        node1feature = np.nonzero(img1.iloc[edge[0]][neighborhoodslist])[0]
        node2feature = np.nonzero(img1.iloc[edge[1]][neighborhoodslist])[0]
        if (len(node1feature)==0) & (img1.iloc[edge[0]]['DAPI/blue'] ==1)& (img1.iloc[edge[0]]['KI67/cyan'] ==0):
            node1feature=[5]
        if (len(node2feature)==0) & (img1.iloc[edge[0]]['DAPI/blue'] ==1) & (img1.iloc[edge[0]]['KI67/cyan'] ==0):
            node2feature=[5]
        for x in node1feature:
            for y in node2feature:
                if x!=y:
                    designedfeature[x,y]+=1
                    designedfeature[y,x]+=1
                else:
                    designedfeature[x,y]+=1
    return designedfeature
    #    break
    
    
###################
# chord diagram
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import numpy as np

LW = 0.3

def polar2xy(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])

def hex2rgb(c):
    return tuple(int(c[i:i+2], 16)/256.0 for i in (1, 3 ,5))

def IdeogramArc(start=0, end=60, radius=1.0, width=0.2, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    inner = radius*(1-width)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(inner, end),
        polar2xy(inner, end) + polar2xy(opt*(1-width), end-0.5*np.pi),
        polar2xy(inner, start) + polar2xy(opt*(1-width), start+0.5*np.pi),
        polar2xy(inner, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.LINETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CLOSEPOLY,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)


def ChordArc(start1=0, end1=60, start2=180, end2=240, radius=1.0, chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi/180.
    end1 *= np.pi/180.
    start2 *= np.pi/180.
    end2 *= np.pi/180.
    opt1 = 4./3. * np.tan((end1-start1)/ 4.) * radius
    opt2 = 4./3. * np.tan((end2-start2)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1+0.5*np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1-0.5*np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2+0.5*np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2-0.5*np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)

def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)

def chordDiagram(X, ax, colors=None, width=0.1, pad=2, chordwidth=0.7):
    """Plot a chord diagram

    Parameters
    ----------
    X :
        flux data, X[i, j] is the flux from i to j
    ax :
        matplotlib `axes` to show the plot
    colors : optional
        user defined colors in rgb format. Use function hex2rgb() to convert hex color to rgb color. Default: d3.js category10
    width : optional
        width/thickness of the ideogram arc
    pad : optional
        gap pad between two neighboring ideogram arcs, unit: degree, default: 2 degree
    chordwidth : optional
        position of the control points for the chords, controlling the shape of the chords
    """
    # X[i, j]:  i -> j
    x = X.sum(axis = 1) # sum over rows
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    if colors is None:
    # use d3.js category10 https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if len(x) > 10:
            print('x is too large! Use x smaller than 10')
        colors = [hex2rgb(colors[i]) for i in range(len(x))]

    # find position for each start and end
    y = x/np.sum(x).astype(float) * (360 - pad*len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start+end)
        #print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(tuple(polar2xy(1.1, 0.5*(start+end)*np.pi/180.)) + (angle,))
        z = (X[i, :]/x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0+z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        IdeogramArc(start=start, end=end, radius=1.0, ax=ax, color=colors[i], width=width)
        start, end = pos[(i,i)]
        selfChordArc(start, end, radius=1.-width, color=colors[i], chordwidth=chordwidth*0.7, ax=ax)
        for j in range(i):
            color = colors[i]
            if X[i, j] > X[j, i]:
                color = colors[j]
            start1, end1 = pos[(i,j)]
            start2, end2 = pos[(j,i)]
            ChordArc(start1, end1, start2, end2,
                     radius=1.-width, color=colors[i], chordwidth=chordwidth, ax=ax)

    #print(nodePos)
    return nodePos

##################################




##Neighborhood and Affinity matrix

def assign_attribution_set1(self_type_id):
    
    """Summary or Description of the Function
    in context of ['CK','CD8','FOXP3','GITK-T','Other']
    Parameters:
    cell_type_id : 
    
    Returns:
    Defined cell attribution

   """
    if (self_type_id in [0,1,2,3]):
        i=0
    elif self_type_id==4:
        i=1
    elif (self_type_id in [5,7,8]):
        i=2
    elif self_type_id == 6:
        i=3
    else:
        i=4
    return i


def build_inter_matrix(df_all,img_id,cell_type_list,cell_type_channel,assign_attribution_set1,type_list):
    """Summary or Description of the Function
    Parameters:
    df_all : 
    img_id : 
    
    Returns:
    cell interaction matrix
   """
    G_img = build_network(df_all,img_id=img_id)
    img1 = df_all[df_all['id_image']==img_id].reset_index()
    selftype = cell_type(img1,cell_type_list,cell_type_channel)
    inter_matrix = np.zeros((len(type_list),len(type_list)))
    for edge in list(G_img.edges):
        i= assign_attribution_set1(selftype[edge[0]])
        j= assign_attribution_set1(selftype[edge[1]])
        if i!=j:
            inter_matrix[i,j]+=1
            inter_matrix[j,i]+=1
        else:
            inter_matrix[i,j]+=1
    nodes = [assign_attribution_set1(selftype[x]) for x in G_img.nodes]
    node_count = [nodes.count(i) for i in range(len(type_list))]
    affinity_matrix = inter_matrix/np.array(node_count)[:,None]

    return inter_matrix,node_count,affinity_matrix




def build_cell_neighbor_matrix(df_all,img_id,cell_type_list,cell_type_channel,cell_type_name):
    """Summary or Description of the Function

    Parameters:
    df_all : 
    img_id : 
    
    Returns:
    neighborhood dataframe

   """
    G_img = build_network(df_all,img_id=img_id)
    img1 = df_all[df_all['id_image']==img_id].reset_index()
    selftype = cell_type(img1,cell_type_list,cell_type_channel)
    neighbormatrix = np.zeros((img1.shape[0],len(cell_type_list)))
    neighbor_id_dic = defaultdict(list)
    for edge in list(G_img.edges):
        if edge[1] not in neighbor_id_dic[edge[0]]:
            neighbor_id_dic[edge[0]].append(edge[1])
            neighbormatrix[edge[0]][selftype[edge[1]]]+=1
        else:
            print('in!!')
        if edge[0] not in neighbor_id_dic[edge[1]]:
            neighbor_id_dic[edge[1]].append(edge[0])
            neighbormatrix[edge[1]][selftype[edge[0]]]+=1
        else:
            print('in!!')
    neighbor_df = pd.DataFrame(data=neighbormatrix,columns=cell_type_name)
    neighbor_df.insert(loc=0, column='self', value=selftype)
    neighbor_df.insert(loc=1, column='KI67', value=img1['KI67/cyan'].values)
    neighbor_df.insert(loc=0, column='imgid', value=np.array([img_id]*img1.shape[0]))
    neighbor_df.insert(loc=0, column='patientid', value=np.array([img1['id_patient'][0]]*img1.shape[0]))
    
    nodes = sorted(list(neighbor_id_dic.keys()))

    neighbor_df_select = neighbor_df.iloc[nodes]
    return neighbor_df_select



def convert_one_image(multichannel_tif_path,rgb_png_path,channel_list,color_list,min_max=15):
	im = Image.open(multichannel_tif_path)
	patient= multichannel_tif_path.split('/')[-2]
	number_of_channels = 7
	width, height = im.width,im.height
	raw_width, raw_height = max([ e[1][2] for e in im.tile]),max([ e[1][3] for e in im.tile])
	im_array = np.zeros([raw_height,raw_width,number_of_channels])
	for i, page in enumerate(ImageSequence.Iterator(im)):
		if i in channel_list:
			raw_width, raw_height = max([ e[1][2] for e in page.tile]),max([ e[1][3] for e in page.tile])
			im.size = raw_width, raw_height
			page_arr = np.array(page)
			page_arr[page_arr<=1]=0
			im_array[:,:,i] = page_arr
	unique_color = sorted(set(color_list))
	new_im_array = np.zeros([raw_height,raw_width,len(unique_color)])
	for i,x in enumerate(color_list):
		index = unique_color.index(x)
		new_im_array[:,:,index]= new_im_array[:,:,index]+im_array[:,:,i]
	color_matrix =  np.array([[0,255,0],[255,255,255],[0,255,255],[255,0,255],[255,255,0],[0,0,255],[255,0,0]])
	color_matrix = color_matrix[unique_color]
	image73=(new_im_array/np.array( [max(min_max,x*0.8) for x in np.max(new_im_array,axis=(0,1))])).dot(color_matrix) 
	im = Image.fromarray((np.clip(image73,0,255))[:height,:width].astype('uint8')).convert('RGB')
	outputname = rgb_png_path.replace('tif','png')
	im.save(outputname)
    
    
    
def threshold_one_plot(tiff_arr,index,scale='p98',min_thold=4,max_thold=8,dapi=False):
    # Gaussian blur, Tri-angel
    plt.figure(figsize=(35,7),dpi=100,facecolor='white')
    fig_num=3
    page = ndimage.gaussian_filter(ndimage.median_filter(tiff_arr[index],size=2),sigma=1)
    page_mean = np.percentile(page,80)
    print(page_mean)
    plt.subplot(1,fig_num,1)
    sns.distplot((page.flatten()),bins=100,color='gray',hist=None,label='Blurred')

    sns.distplot((tiff_arr[:,:,index].flatten()),bins=100,hist=True,label='Original') # ,color='deepskyblue'

    tri_thold =filters.threshold_triangle((page),nbins=50)

    otsu_thold =filters.threshold_otsu((page),nbins=60)

    ori_pixel = (len(page.flatten()))
    if dapi :
        thold = max(otsu_thold/3,2.5)
        thold_label = 'Otsu threshold' 

    else: 
        if page_mean>=5:
            thold_label = 'Otsu threshold'
            thold=otsu_thold
        else:
            thold=tri_thold
            thold_label = 'Tri-angle threshold' 
        thold=max(thold,min_thold)
        thold=min(thold,max_thold)


    print('Threshold is: ',thold)


#         after_db = page[page>db_tri_thold].flatten()
    signal = page[page>thold].flatten()
    signal_rate = len(signal)/ori_pixel
#         after_db_rate = len(after_db)/ori_pixel
    plt.axvline(x=thold,c='red',label=thold_label + str(signal_rate*100)[:4] + '%')
    #plt.axvline(x=db_tri_thold,c='yellow',label='Double'  + str(after_db_rate*100)[:4] + '%')
    plt.xlabel('Signal intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.subplot(1,fig_num,2)
    plt.imshow(tiff_arr[index],cmap='gray')

    page[(page)<= thold]=0

# No.1

    if scale == 'p98':
        p2, p98 = np.percentile(page, (2, 98))
        img_rescale = exposure.rescale_intensity(page, in_range=(p2, p98))
    elif scale == 'hist':
        img_rescale = exposure.equalize_hist(page)
    elif scale == 'adap':
# Method 4
        if np.max(page)!=0:
            img_rescale = exposure.equalize_adapthist(page/np.max(page),kernel_size=(335,251), clip_limit=0.03)
        else:
            img_rescale = page
    else:    

        img_rescale=page #/np.max(db_tri_thold)

    plt.subplot(1,fig_num,3)
    plt.imshow((img_rescale),cmap='gray')

    plt.show()
    return page


def threshold_one(tiff_arr,index,min_thold=4,max_thold=8,dapi=False):
    # Gaussian blur, Tri-angel
    page = ndimage.gaussian_filter(ndimage.median_filter(tiff_arr[index],size=2),sigma=1)
    page_mean = np.percentile(page,80)
    tri_thold =filters.threshold_triangle((page),nbins=50)
    otsu_thold =filters.threshold_otsu((page),nbins=60)
    ori_pixel = (len(page.flatten()))
    if dapi:
        thold = max(otsu_thold/3,2.5)
        thold_label = 'Otsu threshold' 
        thold=max(thold,min_thold)
        thold=min(thold,max_thold)
    else: 
        if page_mean>=5:
            thold_label = 'Otsu threshold'
            thold=otsu_thold
        else:
            thold=tri_thold
            thold_label = 'Tri-angle threshold' 
        thold=max(thold,min_thold)
        thold=min(thold,max_thold)
    return thold





def plot_max_tiff_img_dis(max_arr,bins=100,channel_name= ['chan1', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7'] ,title='Distribution of Max Intensity among all the images'):
    #ranking = []
    plt.figure(figsize=(24,10),dpi=70,facecolor='white')
    channel_num=len(channel_name)
    for channel in range(channel_num):
        plt.subplot(2,4,channel+1)
        sns.distplot(max_arr[:,channel],bins=bins)
        plt.title(channel_name[channel])    
    plt.suptitle(title,fontsize=25)
    plt.show()
    
    
## QC REPORT
def one_image(file_path,sample_name,channel_name,file_size,rank=0):
    img_arr = tifffile.imread(file_path)
    channel_name  =channel_name + ['Inversed Composite']
    #plt.figure(figsize=(30,10))
    for i in range(8):
        if i==0:
            plt.subplot(2*file_size,4,(i+1)+10*(rank))
            plt.axis('off')
            plt.text(0.5,0.5, ('Index:%d\nsample ID:%s\nImage ID: %s\n') % (rank,sample_name,file_path.replace('[',']').split(']')[1]),ha='center',va='center',fontsize=20)
            continue
        open_image=img_arr[i-1]
        plt.subplot(2*file_size,4,(i+1)+10*(rank))
        plt.imshow(open_image)
        plt.colorbar()
        plt.title('Channel %d: %s' % (i-1,channel_name[i-1]))
        
def one_report(img_paths,save_dir,project_name,channel_name,show=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_list = img_paths
    file_size = len(file_list)
    dataID = img_paths[0].split('/')[-1].split(' ')[0]
    #print ('Processing ' + folder)
    index = 0
    plt.figure(figsize=(30,10*file_size),dpi=80)
    for rank,file in enumerate(file_list):
        one_image(file,channel_name=channel_name,sample_name=dataID,file_size=file_size,rank=rank)
        index+=1
    plt.suptitle('\n\nQuality control of Vectra data: '+dataID,fontsize=50,y=0.99)
    plt.tight_layout(rect=[0.01, 0.03, 0.97, 1-0.5/file_size])
    plt.savefig(os.path.join(save_dir,project_name+'QC-report-'+dataID + '-before.png'))
    if show: plt.show()
    plt.close()

def one_image_norm_clean(file_path,dataID,sample_name,channel_name,file_size,rank,thold,cap):
    img_arr1 = tifffile.imread(file_path)
    channel_name  =channel_name + ['Inversed Composite', 'Colored Composite']
    #plt.figure(figsize=(30,10))
    img_arr2 =  tifffile.imread(file_path)
    row=4
    for i in range(row*4):
        if i<row*2:
#             if i==8:
#                 continue
            if i==0:
                plt.subplot(row*file_size,row,(i+1)+4*row*(rank))
                plt.axis('off')
                plt.text(0.5,0.5, ('Original\nIndex:%d\nsample ID:%s\nImage ID: %s\n') % (rank,sample_name,file_path.replace('[',']').split(']')[1]),ha='center',va='center',fontsize=20)
                continue
            open_image=img_arr1[i-1]
            plt.subplot(row*file_size,row,(i+1)+4*row*(rank))
            plt.imshow(open_image)
            plt.colorbar()
            plt.title('Channel %d: %s' % (i-1,channel_name[i-1]))
            
        
        else:
            
            if i==2*row:
                plt.subplot(row*file_size,row,(i+1)+row*4*(rank))
                plt.axis('off')
                plt.text(0.5,0.5, ('Clean\nIndex:%d\nsample ID:%s\nImage ID: %s\n') % (rank,sample_name,file_path.replace('[',']').split(']')[1]),ha='center',va='center',fontsize=20)
                continue
            open_image_2 = img_arr1[i-9]
            plt.subplot(row*file_size,row,(i+1)+row*4*(rank))
            page=np.array(open_image_2.copy())
            page = ndimage.gaussian_filter(ndimage.median_filter(page,size=2),sigma=1)
            if i<row*4:
                page[page<thold[i-9]]=0
                #page = np.ndarray.clip(page,min=0,max=cap[i-11])
                page = np.ndarray.clip(page/cap[i-9],min=0,max=1)
            plt.imshow(page)
            plt.colorbar()
            #max2 = np.clip(np.max(ndimage.gaussian_filter(open_image),sigma=1),a_min=10,a_max=50)
            #plt.clim()
            plt.title('Channel %d: %s' % (i-11,channel_name[i-9]))
            
def one_report_clean(id_,img_paths,save_dir,sample_name,project_name,thod_df,cap_df,channel_name,show=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_list = img_paths
    file_size = len(file_list)
    dataID = img_paths[0].split('/')[-1].split(' ')[0]
    #print ('Processing ' + folder)
    index = 0
    plt.figure(figsize=(24,25*file_size),dpi=60)
    for rank,file in enumerate(file_list):
        one_image_norm_clean(file,dataID,sample_name=sample_name,channel_name=channel_name,file_size=file_size,rank=rank,thold=thod_df.iloc[id_[rank],:],cap=cap_df.loc[sample_name,:])
        index+=1
    plt.suptitle('\n\nQuality control of Vectra data: '+dataID,fontsize=50,y=0.99)
    #plt.tight_layout(rect=[0.01, 0.03, 0.97, 1-0.4/file_size])
    plt.savefig(os.path.join(save_dir,project_name+'QC-report-'+dataID + '-after.jpg'))
    if show: plt.show()
    plt.close()
    
def image_pre_process(img_arr,channel,thold,cap):
    page = ndimage.gaussian_filter(ndimage.median_filter(img_arr[channel],size=2),sigma=1)
    page[page<thold]=0
    page = np.ndarray.clip(page/cap,min=0,max=1)
    return page
    
    
def CM_generator_image(tiff_images,rle_images,id_image,marker_list,list_sample,unique_sample,sample_max_df,sample_min_df):
    df_header =  ['id_image','id_sample','id_cell','area','centroid_x','centroid_y','orientation','eccentricity','minor_axis_length','major_axis_length'] + sum([[x +'_sum',x +'_area' ] for x in marker_list],[])
    matrix_count = []
    name_sample = list_sample[id_image]
    id_sample = list(unique_sample).index(name_sample)
    #print(name_sample)
    #mask=rle2mask(rle_images[id_image])
    cell_num = sum(1 for i in open(rle_images[id_image], 'rb')) -1 
    OpenRLE = open(rle_images[id_image],'r')
    mask_shape = [int(x) for x in OpenRLE.readline().split(',')]
    
    arr_image = tifffile.imread(tiff_images[id_image])[:len(marker_list)]
    image_shape = arr_image.shape[1:]
    #list_clean_image = []
    list_clean_image = arr_image.copy()
    matrix_df = pd.DataFrame(0,index=range(cell_num),columns=df_header)
    for channel in range(len(marker_list)):
        clean_image = image_pre_process(arr_image,channel=channel,cap=sample_max_df.loc[name_sample][channel],thold=sample_min_df.iloc[id_image,channel])
        list_clean_image[channel]=clean_image

    #for local_id,cell in enumerate(mask):
    for local_id,line in enumerate(OpenRLE):
        cell = rle_decode(' '.join(line.split(',')[1].split(' ')[1:]),mask_shape)
        cell =cell[:image_shape[0],:image_shape[1]]
        #list_feature = []
        matrix_df.loc[local_id,'id_image'] = id_image
        matrix_df.loc[local_id,'id_sample'] = id_sample
        matrix_df.loc[local_id,'id_cell'] = local_id
# Size filter is added into segmentation part
#             if np.sum(cell)<10:
#                 continue
#         try:
        props = regionprops(1*cell)[0]
#         except:
#             print(np.sum(cell),'error')
#             continue
        matrix_df.loc[local_id,'area'] = props.area   
        matrix_df.loc[local_id,'centroid_x'] = props.centroid[0]
        matrix_df.loc[local_id,'centroid_y'] = props.centroid[1] 
        matrix_df.loc[local_id,'orientation'] = props.orientation
        matrix_df.loc[local_id,'eccentricity'] = props.eccentricity
        matrix_df.loc[local_id,'minor_axis_length'] = props.minor_axis_length
        matrix_df.loc[local_id,'major_axis_length'] = props.major_axis_length
        for channel in range(7):   
            matrix_df.loc[local_id,df_header[10+channel*2]] = np.sum(list_clean_image[channel][cell])
            matrix_df.loc[local_id,df_header[11+channel*2]] = np.sum(list_clean_image[channel][cell]>0)
        #print(matrix_df)
        #break
    OpenRLE.close()
    return matrix_df
            
            
        
def plot_marker_distribution(data,title,log=False):
    fig_num = data.shape[1]
    col_num = 4
    row_num = fig_num //col_num +1
    plt.figure(figsize=(4*col_num,4*row_num),dpi=60)
    for i in range(fig_num):
        plt.subplot(row_num,col_num,i+1)
        if log:
            sns.distplot(np.log10(data.iloc[:,i]+1),bins=50,hist_kws={'log':False},norm_hist=True)
        else:
            sns.distplot((data.iloc[:,i]),bins=50,hist_kws={'log':False},norm_hist=True)
        #ax.set_yscale('log')
    if log:
        title=title+'_Log10'
    plt.suptitle(title)
    #plt.tight_layout()
    plt.show()


def kde(x, y):
    xmin=min(x)
    xmax=max(x)
    ymin=min(y)
    ymax=max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    density = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, density


def plot_pair_distribution(dataframe,height=5,sample_size=10000,n_contour=12,dpi=120,scatter_size=20,cmap='hot'):
    nrow=ncol = dataframe.shape[1]
    plt.figure(figsize=(nrow*height,ncol*height),dpi=dpi)
    for i in range(nrow):
        for j in range(ncol):
            if i<=j:
                continue
            else:
                sub_df=dataframe[(dataframe.iloc[:,i]>0) & (dataframe.iloc[:,j]>0)]
                sub_df= sub_df.sample(n=sample_size,random_state=101,replace=1)
                x=sub_df.iloc[:,i].values
                y=sub_df.iloc[:,j].values
                xy=np.vstack([x,y])
                plt.subplot(nrow,ncol,i*nrow+j+1)
                sns.scatterplot(x=dataframe.columns[i],y=dataframe.columns[j],data=sub_df,s=scatter_size)
                xx,yy,density=kde((x), (y))
                plt.contour(xx,yy,np.arcsinh(density),n_contour,cmap=cmap)
    plt.show()




