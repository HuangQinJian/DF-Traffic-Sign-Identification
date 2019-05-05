#!/usr/bin/env python
# coding=UTF-8
'''
@Description:
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-05-01 12:36:04
@LastEditTime: 2019-05-04 15:29:01
'''
import os

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from tqdm import tqdm

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12


config_file = "../configs/e2e_faster_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


def load(img_path):
    pil_image = Image.open(img_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


coco_demo = COCODemo(
    cfg,
    min_image_size=1600,
    confidence_threshold=0.7,
)

imgs_dir = '/data1/hqj/traffic-sign-identification/test'
img_names = os.listdir(imgs_dir)

submit_v4 = pd.DataFrame()
empty_v4 = pd.DataFrame()

filenameList = []

X1List = []
X2List = []
X3List = []
X4List = []

Y1List = []
Y2List = []
Y3List = []
Y4List = []

TypeList = []

empty_img_name = []

# for img_name in img_names:
for i, img_name in enumerate(tqdm(img_names)):
    path = os.path.join(imgs_dir, img_name)
    image = load(path)
    # compute predictions
    predictions = coco_demo.compute_prediction(image)
    try:
        scores = predictions.get_field("scores").numpy()
        bbox = predictions.bbox[np.argmax(scores)].numpy()
        labelList = predictions.get_field("labels").numpy()
        label = labelList[np.argmax(scores)]
        filenameList.append(img_name)
        X1List.append(round(bbox[0]))
        Y1List.append(round(bbox[1]))
        X2List.append(round(bbox[2]))
        Y2List.append(round(bbox[1]))
        X3List.append(round(bbox[2]))
        Y3List.append(round(bbox[3]))
        X4List.append(round(bbox[0]))
        Y4List.append(round(bbox[3]))
        TypeList.append(label)
        # print(filenameList, X1List, X2List, X3List, X4List, Y1List,
        #       Y2List, Y3List, Y4List, TypeList)
        print(label)
    except:
        empty_img_name.append(img_name)
        print(empty_img_name)

submit_v4['filename'] = filenameList
submit_v4['X1'] = X1List
submit_v4['Y1'] = Y1List
submit_v4['X2'] = X2List
submit_v4['Y2'] = Y2List
submit_v4['X3'] = X3List
submit_v4['Y3'] = Y3List
submit_v4['X4'] = X4List
submit_v4['Y4'] = Y4List
submit_v4['type'] = TypeList

empty_v4['filename'] = empty_img_name

submit_v4.to_csv('submit_v4.csv', index=None)
empty_v4.to_csv('empty_v4.csv', index=None)
