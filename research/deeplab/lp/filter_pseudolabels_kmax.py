#!/usr/bin/python3

import os
import sys
import glob
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

from tqdm import tqdm

# sys.path.append('/c/Users/mdous/Repo/personal/graph-label-propagation')
# Need to have graph-label-propagation in PYTHONPATH to save the images
import utils.save_annotation as voc_save
from utils import get_dataset_colormap

import argparse

SLIC_DATA_FORMAT = '_slic.npz.npy'
BACKGROUND_LABEL = 0
TYPE_CHOICES = ('abs', 'perc')

def create_annotation(labels_filt, slicpath):
    slicdata = np.load(os.path.join(slicpath, imfile + SLIC_DATA_FORMAT))
    annotation = np.full(np.shape(slicdata), 255, dtype=np.uint8)
    for i in range(labels_filt.shape[0]):
        sp = labels_filt[i,0]
        label = labels_filt[i,1]
        annotation[np.nonzero(slicdata==sp)] = label
    return annotation

def filter_labels(labels, confidences, K = 60, thres = 0.5, filt_type = 'perc'):
    valid_sp = []
    labels_present = np.unique(labels)

    labels_filt = np.empty((0,2), dtype=labels.dtype)
    for l in labels_present:
        superpixs = np.nonzero(labels == l)[0]
        confs = confidences[superpixs]
        val_confs = confs >= thres
        confs = confs[val_confs]
        superpixs = superpixs[val_confs]
        if l == BACKGROUND_LABEL:
            if filt_type not in TYPE_CHOICES:
                raise ValueError("Invalid filter type")
            elif filt_type == 'perc':
                N = np.size(superpixs)
                Keff = int(K*N/100)
            elif filt_type == 'abs':
                Keff = K
            ind = confs.argsort()[-Keff:]
        else:
            ind = range(np.size(superpixs))
        
        tmp_labels = superpixs[ind]
        rep_label = np.full(tmp_labels.size, l, dtype=np.uint8)
        tmp_labels = np.column_stack((tmp_labels, rep_label))
        
        labels_filt = np.concatenate((labels_filt, tmp_labels))

    return labels_filt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter pseudolabels.")
    
    parser.add_argument('--labelpath', type=str, help='Path to pseudolabels file')
    parser.add_argument('--confidencepath', type=str, help='Path to confidences file')
    parser.add_argument('--slicpath', type=str, help='Path to SLIC data folder')
    parser.add_argument('--outputpath', type=str, help='Output folder', default='./')
    # parser.add_argument('--scribblepath', type=str, help='Path to the scribbles (images)')

    parser.add_argument('--K', type=int, help='Percentage or absolute number of superpixels to keep (see type)', default=60)
    parser.add_argument('--type', type=str, help='Choose between absolute and percentage modes', default='perc', choices=TYPE_CHOICES)
    parser.add_argument('--threshold', type=float, help='Minimum confidence level', default=0.95)
    parser.add_argument('--imagepath', type=str, help='Path to original VOC images (for visualization)' ,default='')
    parser.add_argument('--vis', dest='visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--save-overlays', dest='save_overlays', action='store_true', help='Save overlays')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', help='Do not save images')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
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

        labels_filt = filter_labels(labels, confidence, **configuration)
        annotation = create_annotation(labels_filt, args.slicpath)
    
        if args.visualize:
            from PIL import Image
            impath = os.path.join(args.imagepath , imfile + '.jpg')
            im = Image.open(impath)
            np_im = np.array(im, dtype=np.uint8)

            plt.imshow(np_im)
            plt.imshow(annotation, alpha=0.7, cmap=pascal_cmap, vmin=0, vmax=255)
            plt.axis('off')
            if args.save_overlays:
                plt.savefig(os.path.join(OUTPUT_PATH, '{}_overlay.png'.format(imfile)))
            plt.draw()
        
        if not args.dry_run:
            voc_save.save_annotation_indexed(annotation, OUTPUT_PATH, imfile)
