# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import sys
import glob
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2


# %%
sys.path.append('/c/Users/mdous/Repo/personal/graph-label-propagation')
import utils.save_annotation as voc_save
from utils import get_dataset_colormap


# %%
flags = {'vis': True}
if flags['vis']:
    pascal_palette = get_dataset_colormap.create_pascal_label_colormap(256)
    pascal_palette[-1,:] = [255,255,255]
    pascal_palette = np.squeeze(pascal_palette.reshape((-1,1)))
    pascal_palette = pascal_palette.reshape((-1,3))/255
    print(pascal_palette.shape)
    pascal_cmap = colors.ListedColormap(pascal_palette, name = 'pascal', N=256)

# %%
BASE_PATH = '/d/Data/lp_data'
IMAGE_PATH = '/c/Users/mdous/Datasets/VOCdevkit/VOC2012/JPEGImages'
DATA_PATH = os.path.join(BASE_PATH, 'lp_full_500_20_1.0_0.0_1.0_1.0_True_0.95_0_False_0.01_500_0.5_512_dummy_False_0')
SLIC_PATH = os.path.join(DATA_PATH, 'fig/slic')
SCRIBBLE_PATH = os.path.join(BASE_PATH, 'scribble')
CONFIDENCE_PATH = os.path.join(DATA_PATH, 'Confidences')
LABEL_PATH = os.path.join(DATA_PATH, 'fig/pseudomask')
OUTPUT_PATH = os.path.join(DATA_PATH,'fig/pm_filt_test')
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
get_filename = lambda x: os.path.splitext(os.path.basename(x))[0]


# %% 
def scribble_mask(imfile, dist=10, maskSize=5):
    im = Image.open(imfile)
    np_im = np.array(im, dtype=np.uint8)
    # scribble_pos = np.asarray(np_im < 255).nonzero()
    np_scribble = np.where(np_im < 255, 0, 255).astype(np.uint8)
    chamf = cv2.distanceTransform(np_scribble, cv2.DIST_C, maskSize)
    mask_d = np.where(chamf > dist, False, True)
    return mask_d


# %%
def valid_superpixels(imfile, mask_d):
    # find superpixels containing the scribbles
    slpath = os.path.join(SLIC_PATH, get_filename(imfile) + '_slic.npz.npy')
    slic_im = np.load(slpath)    
    superpixs = np.unique(slic_im[mask_d])  
    mask_sp = np.full(slic_im.shape, False, dtype=bool)
    for sp in superpixs:
        mask_sp[slic_im == sp] = True
    return mask_sp


# %%
def confidence_mask(imfile, threshold=0.9):
    conffile = os.path.join(CONFIDENCE_PATH, os.path.basename(imfile))
    confidence = Image.open(conffile)
    np_conf = np.array(confidence, dtype=np.float)/65535.
    mask_conf = np.where(np_conf > threshold, True, False)
    return mask_conf


# %%
def filter_label(imfile, mask):
    labelfile = os.path.join(LABEL_PATH, get_filename(imfile) + '_pseudomask.png')
    label = Image.open(labelfile)
    np_label = np.array(label)
    filtered_label = np_label*mask
    return filtered_label

# %%
if __name__ == "__main__":
    filelist = glob.glob(CONFIDENCE_PATH + "/*.png")
    for imfile in filelist:
        mask = confidence_mask(imfile) & valid_superpixels(imfile, scribble_mask(imfile))
        filtered_label = filter_label(imfile, mask)
        
        if flags['vis']:
            impath = os.path.join(IMAGE_PATH, get_filename(imfile) + '.jpg')
            im = Image.open(impath)
            np_im = np.array(im, dtype=np.uint8)
            plt.subplot(1,2,1)
            #plt.imshow(np_im)
            plt.imshow(filtered_label, cmap=pascal_cmap, alpha=1.)
            
            plt.subplot(1,2,2)
            plt.imshow(mask)
            plt.show()
        
        # voc_save.save_annotation_indexed(filtered_label, OUTPUT_PATH, get_filename(imfile))


# %%
