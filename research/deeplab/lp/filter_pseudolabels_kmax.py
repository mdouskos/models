#!/usr/bin/python3

import os
import sys
import glob
import pickle
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import colors

from tqdm import tqdm

from scipy.spatial import distance

# sys.path.append('/c/Users/mdous/Repo/personal/graph-label-propagation')
# Need to have graph-label-propagation in PYTHONPATH to save the images
import utils.save_annotation as voc_save
from utils import get_dataset_colormap

import argparse

SCRIBBLE_FORMAT = '.png'
IMAGE_FORMAT = '.jpg'
OVERLAY_FORMAT = '_overlay.png'
SLIC_DATA_FORMAT = '_slic.npz.npy'

BACKGROUND_LABEL = 0
TYPE_CHOICES = ('relative', 'percentage', 'absolute')

def create_annotation(labels_filt, slicdata):
    annotation = np.full(np.shape(slicdata), 255, dtype=np.uint8)
    for i in range(labels_filt.shape[0]):
        sp = labels_filt[i,0]
        label = labels_filt[i,1]
        annotation[np.nonzero(slicdata==sp)] = label
    return annotation

def filter_labels(labels, confidences, scribble, K = 60, thres = 0.5, filt_type = 'percentage'):    
    valid_sp = []
    labels_present = np.unique(labels)

    labels_filt = np.empty((0,2), dtype=labels.dtype)
    for l in labels_present:
        superpixs = np.nonzero(labels == l)[0]
        confs = confidences[superpixs]
        val_confs = confs >= thres
        confs = confs[val_confs]
        superpixs = superpixs[val_confs]

        
        N = np.size(superpixs)
        scribble_ind = np.argwhere(scribble==l)
        scribble_sp_dist_fun = lambda x: distance.cdist(scribble_ind, np.argwhere(slicdata==x)).min()
        scribble_dist_map = map(scribble_sp_dist_fun, superpixs)
        scribble_dist_array = np.fromiter(scribble_dist_map, dtype=np.float32)
        sp_sort_ind = np.argsort(scribble_dist_array)
        S = np.sum(scribble_dist_array==0)
        if filt_type not in TYPE_CHOICES:
            raise ValueError("Invalid filter type")
        elif filt_type == 'percentage':
            lim = int(K*N)
        elif filt_type == 'relative':
            lim = int(K*S)
        elif filt_type == 'absolute':
            lim = int(K+S)

        # if l == BACKGROUND_LABEL:
        #   ind = confs.argsort()[-Keff:]
        # else:
        #   ind = range(np.size(superpixs))

        ind = sp_sort_ind[:lim]
        tmp_labels = superpixs[ind]
        rep_label = np.full(tmp_labels.size, l, dtype=np.uint8)
        tmp_labels = np.column_stack((tmp_labels, rep_label))
        
        labels_filt = np.concatenate((labels_filt, tmp_labels))

    return labels_filt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter pseudolabels.")
    
    parser.add_argument('--labelpath', type=str, help='Path to pseudolabels file', required=True)
    parser.add_argument('--confidencepath', type=str, help='Path to confidences file', required=True)
    parser.add_argument('--slicpath', type=str, help='Path to SLIC data folder', required=True)
    parser.add_argument('--scribblepath', type=str, help='Path to scribble folder', required=True)
    parser.add_argument('--outputpath', type=str, help='Output folder', default='./', required=True)
    # parser.add_argument('--scribblepath', type=str, help='Path to the scribbles (images)')

    parser.add_argument('--K', type=float, help='Percentage or absolute number of superpixels to keep (see type)', default=2)
    parser.add_argument('--type', type=str, help='''
            Choose superpixel selection modes (for each label).
            Absolute: keeps K superpixels;
            Relative: keeps K*S superpixels with S number of superpixels containing scribble;
            Percentage: keeps K*N superpixels with N total number of superpixels.
            ''', default='relative', choices=TYPE_CHOICES)
    parser.add_argument('--threshold', type=float, help='Minimum confidence level', default=0.95)
    parser.add_argument('--imagepath', type=str, help='Path to original VOC images (for visualization)' ,default='')
    parser.add_argument('--vis', dest='visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--save-overlays', dest='save_overlays', action='store_true', help='Save overlays')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', help='Do not save images')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.visualize:
        pascal_palette = get_dataset_colormap.create_pascal_label_colormap(256)
        pascal_palette[-1,:] = [255,255,255] # fix ignore color
        pascal_palette = np.squeeze(pascal_palette.reshape((-1,1)))
        pascal_palette = pascal_palette.reshape((-1,3))/255
        pascal_cmap = colors.ListedColormap(pascal_palette, name = 'pascal', N=256)

    with open(args.confidencepath, 'rb') as f:
        datasetconf = pickle.load(f)

    with open(args.labelpath, 'rb') as f:
        datasetlabels = pickle.load(f)
    
    OUTPUT_PATH = args.outputpath
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    filelist = datasetlabels.keys() 
    for imfile in tqdm(filelist):
        labels = datasetlabels[imfile]
        confidence = datasetconf[imfile]

        configuration = {'K': args.K,
                         'thres': args.threshold,
                         'filt_type': args.type}

        scribbleim = Image.open(os.path.join(args.scribblepath, imfile + SCRIBBLE_FORMAT))
        scribble = np.array(scribbleim, dtype=np.uint8)

        slicdata = np.load(os.path.join(args.slicpath, imfile + SLIC_DATA_FORMAT))

        labels_filt = filter_labels(labels, confidence, scribble, **configuration)
        annotation = create_annotation(labels_filt, slicdata)
    
        if args.visualize:
            impath = os.path.join(args.imagepath , imfile + IMAGE_FORMAT)
            im = Image.open(impath)
            np_im = np.array(im, dtype=np.uint8)

            plt.imshow(np_im)
            plt.imshow(scribble, alpha=0.5, cmap=pascal_cmap, vmin=0, vmax=255)
            plt.imshow(annotation, alpha=0.5, cmap=pascal_cmap, vmin=0, vmax=255)
            plt.axis('off')
            if args.save_overlays:
                plt.savefig(os.path.join(OUTPUT_PATH, '{}{}'.format(imfile, OVERLAY_FORMAT)))
            plt.show()
        
        if not args.dry_run:
            voc_save.save_annotation_indexed(annotation, OUTPUT_PATH, imfile)
