#! /usr/bin/env python3

# Copyright 2017 Intel Corporation. 
# The source code, information and material ("Material") contained herein is  
# owned by Intel Corporation or its suppliers or licensors, and title to such  
# Material remains with Intel Corporation or its suppliers or licensors.  
# The Material contains proprietary information of Intel or its suppliers and  
# licensors. The Material is protected by worldwide copyright laws and treaty  
# provisions.  
# No part of the Material may be used, copied, reproduced, modified, published,  
# uploaded, posted, transmitted, distributed or disclosed in any way without  
# Intel's prior express written permission. No license under any patent,  
# copyright or other intellectual property rights in the Material is granted to  
# or conferred upon you, either expressly, by implication, inducement, estoppel  
# or otherwise.  
# Any license under such intellectual property rights must be express and  
# approved by Intel in writing.

from mvnc import mvncapi as mvnc
import numpy
import cv2
import os, sys
import KMeans_Watershed_1 as KW

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]
#subimages, number = KW.main(image_path)
subimages = cv2.imread(image_path)
number = 1

path_to_networks = './Inception-v3/'
#path_to_images = dir
graph_filename = 'graph'

# mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

# Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

# Load preprocessing data
mean = 128
std = 1 / 128

# Load categories
categories = []
with open(path_to_networks + 'categories.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

# Load image size
'''
with open(path_to_networks + 'inputsize.txt', 'r') as f:
    reqsize = int(f.readline().split('\n')[0])
'''
reqsize = 299

graph = device.AllocateGraph(graphfile)

for k in range(number):
    print("Subimage %s"%k)
    image=subimages
    img = image

    img = cv2.resize(img, (reqsize, reqsize))

    cv2.imshow('%s' % k, img)

    print('Start download to NCS...')
    graph.LoadTensor(img.astype(numpy.float16), 'user object')
    output, userobj = graph.GetResult()

    top_inds = output.argsort()[::-1][:5]

    print(''.join(['*' for i in range(79)]))
    print('inception-v3 on NCS')
    print(''.join(['*' for i in range(79)]))
    for i in range(5):
        print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

    print(''.join(['*' for i in range(79)]))

graph.DeallocateGraph()
device.CloseDevice()
print('Finished')
