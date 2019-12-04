#!/usr/bin/env python
# coding: utf-8

# # Anchor Boxes Analysis using K-Means


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns


#设置风格，尺度
sns.set_style('darkgrid')
sns.set_context('paper')

# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import cv2
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

import matplotlib.patches as patches
# import scipy.stats as sci

# 对照的生成函数=======================

#stride基础框大小        默认 ANCHOR_STRIDE: 8，16，32
#sizes目标框大小         默认 ANCHOR_SIZE: 128, 256, 512
#ratios高宽比           默认 ANCHOR_RATIOS: 0.5, 1.0, 2.0
#==================================


def _compute_losses(sides, input_sides):
    loss=0
    for side in sides:
        length = []
        for input_side in input_sides:
            length.append((input_side-side)**2)
        loss += min(length)
        
    return loss



# function from Tensorflow Object Detection API to resize image
def _compute_new_static_size(width, height, min_dimension, max_dimension):
    orig_height = height
    orig_width = width
    orig_min_dim = min(orig_height, orig_width)
  
    # Calculates the larger of the possible sizes
    large_scale_factor = min_dimension / float(orig_min_dim)
      # Scaling orig_(height|width) by large_scale_factor will make the smaller
      # dimension equal to min_dimension, save for floating point rounding errors.
      # For reasonably-sized images, taking the nearest integer will reliably
      # eliminate this error.
    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]
    if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]
        new_size = large_size
    if max(large_size) > max_dimension:
        new_size = small_size
    else:
        new_size = large_size
    
    return new_size[1], new_size[0]


# ### IOU based clusterring


#utility functions for K-means
import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters



# #### Read csv
# A csv was made from xml file using "xml_to_csv.py"


#Read the dataset
data = pd.read_csv('my_labels.csv')

#utility function to get width & height
def change_to_wh (data):
    data['w'] = data['xmax'] - data['xmin'] + 1
    data['h'] = data['ymax'] - data['ymin'] + 1
    return data

min_dimension = 720
max_dimension = 1280


# Initial Data
data.describe()

data = change_to_wh(data)
data['new_w'], data['new_h'] = np.vectorize(_compute_new_static_size)(data['width'], 
                                                                      data['height'], min_dimension, max_dimension)

data['b_w'] = data['new_w']*data['w']/data['width']
data['b_h'] = data['new_h']*data['h']/data['height']
data['b_ar'] = data['b_h']/data['b_w']

# ### Calculate the base box size!

'''
input_array=[16,32,64,128,256,512,1024]
bound_array=[11,22,45,90,181,362,724,1448]

def count_base_size(width, height, input_array=[16,32,64,128,256,512,1024]):
    result = {}
    for ele in input_array:
        result[str(ele)] = 0
    result['rest'] = 0
    
    for w,h in zip(width,height):
        done = False
        for inp in input_array:
            if w <= inp and h <= inp:
                result[str(inp)] += 1
                done = True
        if done == False:
            result['rest'] += 1
            
    return result
    
D = count_base_size(data["b_w"].tolist(), data["b_h"].tolist(), bound_array)

print('box 尺寸分布统计')
print(D)


# bound_array=[11,22,45,90,181,362,724,1448]
key_str = ['0-11', '11-22', '22-45', '45-90', '90-181', '181-362', '362-724', '724-1448', '1448+']
# bound_array=[11,22,45,90,181,362,724,1448]
value_list = []
for i in range(0, len(bound_array),1):
    value_list.append(D[str(bound_array[i])])
value_list.append(D['rest'])
# print(value_list)

plt.figure()
plt.bar(range(len(value_list)), value_list, align='center') 
plt.xticks(range(len(key_str)), key_str) 
plt.ion()
plt.show()


# ====================================================

# bound_array=[11,22,45,90,181,362,724,1448]
for i in range(1, len(bound_array),1):
    num1 = D[str(bound_array[i-1])]
    num2 = D[str(bound_array[i])]
    D[str(bound_array[i])] = num2-num1

print('box 尺寸范围分布统计')
print(D)
print('Aspect Sizes:')
print(input_array)
'''

# Area Size
data['b_side_size'] = (data['b_w']*data['b_h']).apply(np.sqrt)
data.describe()


data.columns


# ================= 独立分析 ================
'''
sides = data['b_side_size']

best_idx = 0
best_vlue = 92258024570
kk=5
for ii in range(10,60,1):
    input_side=[]
    for jj in range(kk):
        input_side.append(ii*2**jj)

    print(input_side)
    if max(input_side) > max_dimension:
        break
    
    loss=_compute_losses(sides, input_side)
    print(loss)
    
    if loss < best_vlue:
        best_vlue=loss
        best_idx=ii

print('\nFind The Best:', best_idx, best_vlue)
exit()
'''

# ## 独立分析锚尺寸

K_num = 5
X = data.iloc[:,15].values
X = X.reshape(-1,1)
print('独立分析锚尺寸')
K = KMeans(K_num, random_state=1)
labels = K.fit(X)
print("Anchor Sizes: (k=%d)"%K_num)
out = labels.cluster_centers_
print(sorted(out.tolist()))

#=======分布图=======
plt.figure(figsize = (18, 8))
X = data['b_side_size']
# print(X)
plt.subplot(2,1,1)
ax = sns.distplot(X, bins = 60, hist = True, kde = True, norm_hist = False,
            rug = False, vertical = False,
            color = 'b', label = 'distplot', axlabel = 'x')
ax.set_xticks(np.linspace(0, 1000, 41))
plt.xlim(-25,1025)
#=======聚类图=======
plt.subplot(2,1,2)
Y = np.linspace(0, 0, len(X))
#print(len(X), len(Y))
plt.scatter(X, Y, c=labels.labels_, s=50, cmap='viridis',alpha=0.5);
Y = np.linspace(0, 0, len(out))
plt.scatter(out, Y, s=50,c='Blue',marker='*');
plt.xticks(np.linspace(0, 1000, 41))
plt.xlim(-25,1025)
plt.ioff()
plt.show()
#=======




# Aspect Ratio

K_num = 5
X = data.iloc[:,14].values
X = X.reshape(-1,1)
print('独立分析高宽比')
K = KMeans(K_num, random_state=1)
labels = K.fit(X)
print("Aspect Ratios: (k=%d)"%K_num)
out = labels.cluster_centers_
print(sorted(out.tolist()))

#=======分布图=======
plt.figure(figsize = (18, 8))
X = data['b_ar']
# print(X)
plt.subplot(2,1,1)
ax = sns.distplot(X, bins = 60, hist = True, kde = True, norm_hist = False,
            rug = False, vertical = False,
            color = 'b', label = 'distplot', axlabel = 'x')
ax.set_xticks(np.linspace(0, 9, 37))
plt.xlim(-0.5,9)
#=======聚类图=======
plt.subplot(2,1,2)
Y = np.linspace(0, 0, len(X))
#print(len(X), len(Y))
plt.scatter(X, Y, c=labels.labels_, s=50, cmap='viridis',alpha=0.5);
Y = np.linspace(0, 0, len(out))
plt.scatter(out, Y, s=50,c='Blue',marker='*');
plt.xticks(np.linspace(0, 9, 37))
plt.xlim(-0.5,9)
plt.ioff()
plt.show()
#=======
exit()


exit()

# ## Clusterring using both width & height
print('综合分析高宽比和锚尺寸')
X = data.as_matrix(columns=data.columns[12:14])

K = KMeans(6, random_state=0)
labels = K.fit(X)
out = labels.cluster_centers_
print(out)
ar = out[:,1]/out[:,0]
size = np.sqrt(out[:,1]*out[:,0])
print("Aspect Ratios: ")
print(ar)
print("Anchor Sizes: ")
print(size)


# ========================================
plt.ion()
plt.figure(1)
plt.scatter(data['b_w'], data['b_h'],s=5, cmap='viridis',alpha=0.5)
plt.xlabel(u'b_w')
plt.ylabel(u'b_h')
currentAxis=plt.gca()
rect=patches.Rectangle((0, 0),720,720,linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect)
rect=patches.Rectangle((0, 0),1280,720,linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect)
rect=patches.Rectangle((0, 0),720,1280,linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect)

plt.show()

#=======
# bound_array=[11,22,45,90,181,362,724,1448]
key_str = ['0-11', '11-22', '22-45', '45-90', '90-181', '181-362', '362-724', '724-1448', '1448+']
# bound_array=[11,22,45,90,181,362,724,1448]
value_list = []
for i in range(0, len(bound_array),1):
    value_list.append(D[str(bound_array[i])])
value_list.append(D['rest'])
# print(value_list)
plt.figure()
plt.bar(range(len(value_list)), value_list, align='center') 
plt.xticks(range(len(key_str)), key_str) 
plt.show()


#=======
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels.labels_,
            s=50, cmap='viridis',alpha=0.5);
plt.scatter(out[:, 0], out[:, 1],s=50,c='Blue',marker='*');

plt.ioff()
plt.show()












