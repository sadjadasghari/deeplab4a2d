#!/usr/bin/env python
#  Sadjad Esfeden, sadjad@ece.neu.edu
# 05/20/2016

import numpy as np
import struct

def a2d_classes():
  classes = {#'adult-climbing' : 1, 'adult-crawling' : 2, 'adult-eating': 3, 'adult-jumping': 4, 'adult-rolling': 5, 'adult-running': 6, 'adult-walking': 7, 'adult-none': 8,
              #'baby-climbing' : 9, 'baby-crawling' : 10,	'baby-rolling': 11, 'baby-walking': 12, 'baby-none': 13, 
              #'ball-flying': 14, 'ball-jumping': 15, 'ball-rolling': 16, 'ball-none': 17, 
              #'bird-climbing': 18, 'bird-eating': 19, 'bird-flying': 20, 'bird-jumping': 21, 'bird-rolling': 22, 'bird-walking': 23,'bird-none': 24,
              #'car-flying': 25, 'car-jumping': 26, 'car-rolling': 27, 'car-running': 28, 'car-none': 29,
              #'cat-climbing': 30, 'cat-eating': 31, 'cat-jumping': 32, 'cat-rolling': 33, 'cat-running': 34, 'cat-walking': 35, 'cat-none': 36, 
              #'dog-crawling': 37, 'dog-eating': 38, 'dog-jumping': 39, 'dog-rolling': 40, 'dog-running': 41, 'dog-walking': 42, 'dog-none': 43} 
              'adult' : 1,  'baby'   : 2,  'ball'        : 3,  'bird'         : 4, 
             'car'    : 5,  'cat'       : 6,  'dog'         : 7}

  return classes

def a2d_palette():
  palette = {(  0,   0,   0) : 0 ,
             (52,   1,   1) : 1 ,
             (103,	1,	1): 1,#2 , #1
             (154,	1,	1): 1,#3, 
             (255,	1,	1): 1,#4, 
             (255,	51,	51): 1,#5, 
             (255,	103,	103): 1,#6, 
             (255,	154,	154): 1,#7, 
             (255,	205,	205): 1,#8, 
             (52, 46,   1) : 2,#9 ,
             (103, 92,   1) : 2,#10 ,
             (255, 235,   51) : 2,#11 ,
             (255, 245,   154) : 2,#12 ,
             (255, 250,   205) : 2,#13 ,
             (41,	205,	1) : 3,#14 , #ball flying
             (52,	255,	1) : 3,#15, 
             (92,	255,	51) : 3,#16, 
             (215,	255,	205) : 3,#17, 
             (1,   52, 36) : 4,#18 , #bird-climbing
             (1,   154, 108) : 4,#19 ,
             (1,   205, 143) : 4,#20 ,
             (1,   255, 179) : 4,#21 ,
             (51,   255, 194) : 4,#22 ,
             (154,   255, 225) : 4,#23 ,
             (205,   255, 240) : 4,#24 ,
             (1,   82, 205) : 5,#25 ,
             (1,   103, 255) : 5,#26 ,
             (51,   133, 255) : 5,#27 ,
             (103,   164, 255) : 5,#28 ,
             (205,   225, 255) : 5,#29 ,
             (26,   1, 52) : 6,#30 ,
             (77,   1, 154) : 6,#31 ,
             (128,   1, 255) : 6,#32 ,
             (154,   51, 255) : 6,#33 ,
             (179,   103, 255) : 6,#34 ,
             (205,   154, 255) : 6,#35 ,
             (230,   205, 255) : 6,#36 ,
             (103,   1, 62) : 7,#37 ,
             (154,   1, 92) : 7,#38 ,
             (255,   1, 153) : 7,#39 ,
             (255,	51,	174) : 7,#40 ,
             (255,	103,	194) : 7,#41 ,
             (255,	154,	215) : 7,#42 ,
             (255,	205,	235) : 7}#,#43 }
  return palette
'''{(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }'''

def a2d_palette_invert():
  palette_list = a2d_palette().keys()
  palette = ()
  
  for color in palette_list:
    palette += color

  return palette

def a2d_mean_values():
  return np.array([103.939, 116.779, 123.68], dtype=np.float32)

def strstr(str1, str2):
  if str1.find(str2) != -1:
    return True
  else:
    return False

# Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
# 'GTcls' key is for class segmentation
# 'GTinst' key is for instance segmentation
def mat2png_hariharan(mat_file, key='GTcls'):
  mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
  return mat[key].Segmentation

def convert_segmentation_mat2numpy(mat_file):
  np_segm = load_mat(mat_file)
  return np.rot90(np.fliplr(np.argmax(np_segm, axis=2)))

def load_mat(mat_file, key='data'):
  mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
  return mat[key]

# Python version of script in code/densecrf/my_script/LoadBinFile.m
def load_binary_segmentation(bin_file, dtype='int16'):
  with open(bin_file, 'rb') as bf:
    rows = struct.unpack('i', bf.read(4))[0]
    cols = struct.unpack('i', bf.read(4))[0]
    channels = struct.unpack('i', bf.read(4))[0]

    num_values = rows * cols # expect only one channel in segmentation output
    out = np.zeros(num_values, dtype=np.uint8) # expect only values between 0 and 255

    for i in range(num_values):
      out[i] = np.uint8(struct.unpack('h', bf.read(2))[0])

    return np.rot90(np.fliplr(out.reshape((cols, rows))))

def convert_from_color_segmentation(arr_3d):
  arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
  palette = a2d_palette()

  # slow!
  for i in range(0, arr_3d.shape[0]):
    for j in range(0, arr_3d.shape[1]):
      key = (arr_3d[i,j,0], arr_3d[i,j,1], arr_3d[i,j,2])
      arr_2d[i, j] = palette.get(key, 0) # default value if key was not found is 0

  return arr_2d

def create_lut(class_ids, max_id=256):
  # Index 0 is the first index used in caffe for denoting labels.
  # Therefore, index 0 is considered as default.
  lut = np.zeros(max_id, dtype=np.uint8)

  new_index = 1
  for i in class_ids:
    lut[i] = new_index
    new_index += 1

  return lut

def get_id_classes(classes):
  all_classes = a2d_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes

######################
'''
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
'''
