#!/usr/bin/env python
#  Sadjad Esfeden, sadjad@ece.neu.edu
# 05/20/2016

import numpy as np

def a2d_classes():
  classes = {'adult-climbing' : 1, 'adult-crawling' : 2, 'adult-eating': 3, 'adult-jumping': 4, 'adult-rolling': 5, 'adult-running': 6, 'adult-walking': 7, 'adult-none': 8,
              'baby-climbing' : 9, 'baby-crawling' : 10,	'baby-rolling': 11, 'baby-walking': 12, 'baby-none': 13, 
              'ball-flying': 14, 'ball-jumping': 15, 'ball-rolling': 16, 'ball-none': 17, 
              'bird-climbing': 18, 'bird-eating': 19, 'bird-flying': 20, 'bird-jumping': 21, 'bird-rolling': 22, 'bird-walking': 23,'bird-none': 24,
              'car-flying': 25, 'car-jumping': 26, 'car-rolling': 27, 'car-running': 28, 'car-none': 29,
              'cat-climbing': 30, 'cat-eating': 31, 'cat-jumping': 32, 'cat-rolling': 33, 'cat-running': 34, 'cat-walking': 35, 'cat-none': 36, 
              'dog-crawling': 37, 'dog-eating': 38, 'dog-jumping': 39, 'dog-rolling': 40, 'dog-running': 41, 'dog-walking': 42, 'dog-none': 43} 
              #'adult' : 1,  'baby'   : 2,  'ball'        : 3,  'bird'         : 4, 
             #'car'    : 5,  'cat'       : 6,  'dog'         : 7}

  return classes

def a2d_palette():
  palette = {(  0,   0,   0) : 0 ,
             (52,   1,   1) : 1 ,
             (103,	1,	1): 2 , #1
             (154,	1,	1): 3, 
             (255,	1,	1): 4, 
             (255,	51,	51): 5, 
             (255,	103,	103): 6, 
             (255,	154,	154): 7, 
             (255,	205,	205): 8, 
             (52, 46,   1) : 9 ,
             (103, 92,   1) : 10 ,
             (255, 235,   51) : 11 ,
             (255, 245,   154) : 12 ,
             (255, 250,   205) : 13 ,
             (41,	205,	1) : 14 , #ball flying
             (52,	255,	1) : 15, 
             (92,	255,	51) : 16, 
             (215,	255,	205) : 17, 
             (1,   52, 36) : 18 , #bird-climbing
             (1,   154, 108) : 19 ,
             (1,   205, 143) : 20 ,
             (1,   255, 179) : 21 ,
             (51,   255, 194) : 22 ,
             (154,   255, 225) : 23 ,
             (205,   255, 240) : 24 ,
             (1,   82, 205) : 25 ,
             (1,   103, 255) : 26 ,
             (51,   133, 255) : 27 ,
             (103,   164, 255) : 28 ,
             (205,   225, 255) : 29 ,
             (26,   1, 52) : 30 ,
             (77,   1, 154) : 31 ,
             (128,   1, 255) : 32 ,
             (154,   51, 255) : 33 ,
             (179,   103, 255) : 34 ,
             (205,   154, 255) : 35 ,
             (230,   205, 255) : 36 ,
             (103,   1, 62) : 37 ,
             (154,   1, 92) : 38 ,
             (255,   1, 153) : 39 ,
             (255,	51,	174) : 40 ,
             (255,	103,	194) : 41 ,
             (255,	154,	215) : 42 ,
             (255,	205,	235) : 43 ,
             
             #(1, 21, 52) : 5 ,
             #(26, 1, 52) : 6 ,
             #(52, 1, 31) : 7 }

  return palette

def palette_demo():
  palette_list = a2d_palette().keys()
  palette = ()
  
  for color in palette_list:
    palette += color

  return palette

def convert_from_color_segmentation(arr_3d):
  arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
  palette = a2d_palette()

  # slow!
  for i in range(0, arr_3d.shape[0]):
    for j in range(0, arr_3d.shape[1]):
      key = (arr_3d[i,j,0], arr_3d[i,j,1], arr_3d[i,j,2])
      arr_2d[i, j] = palette.get(key, 0) # default value if key was not found is 0

  return arr_2d

def get_id_classes(classes):
  all_classes = a2d_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes

def strstr(str1, str2):
  if str1.find(str2) != -1:
    return True
  else:
    return False

def create_lut(class_ids, max_id=256):
  # Index 0 is the first index used in caffe for denoting labels.
  # Therefore, index 0 is considered as default.
  lut = np.zeros(max_id, dtype=np.uint8)

  new_index = 1
  for i in class_ids:
    lut[i] = new_index
    new_index += 1

  return lut
