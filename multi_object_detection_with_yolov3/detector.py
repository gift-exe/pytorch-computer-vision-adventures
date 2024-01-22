from __future__ import division
import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from darknet import DarkNet
from util import *

import pickle as pkl
import pandas as pd
import numpy as np
import cv2
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument('--images', dest='images', help='Image / Directory containing images to perform detection upon', default='imgs', type=str)
    parser.add_argument('--det', dest='det', help='Image / Directory to store detection to', default='det', type=str)
    parser.add_argument('--bs', dest='bs', help='Batch Size', default=1)
    parser.add_argument('--confidence', dest='confidence', help='Objects Confidence to filter predictions', default=0.5)
    parser.add_argument('--nms_thresh', dest='nms_thresh', help='NMS Threshold', default=0.4)
    parser.add_argument('--cfg', dest='cfgfile', help='config file', default='yolo-cfg/yolov3.cfg', type=str)
    parser.add_argument('--weights', dest='weightsfile', help='weightsfile', default='yolo-cfg/yolov3.weights', type=str)
    parser.add_argument('--reso', dest='reso', help='Input resolution of the network. Increase to increase accuracy. Decrease to increase speed', default='416', type=str)

    return parser.parse_args()

def load_classes(namefile):
    fp = open(namefile, 'r')
    names = fp.read().split('\n')[:-1]
    return names




args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')

print('loading network .....')
model = DarkNet('./yolov3.cfg')
model.load_weights('./yolov3.weights')
print('network successfully loaded')

model.net_info['height'] = args.reso
inp_dim = int(model.net_info['height'])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

read_dir = time.time()

try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.', images)))
except FileNotFoundError:
    print('No file or directory with the name {}'.format(images))
    exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]



