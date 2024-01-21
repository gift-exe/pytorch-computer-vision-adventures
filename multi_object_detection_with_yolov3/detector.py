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
    parser.add_argument('--cfg', dest='cfgfile', help='config file', default='')