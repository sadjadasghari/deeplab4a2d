 #!/usr/bin/env sh
 # Sadjad A Esfeden, sadjad@ece.neu.edu
 # 05/09/2016
 
 IMG_PATH="/home/sasghariesfeden/Documents/git/train-DeepLab/exper/voc12/data/images_orig/"
 
 CHAIR="${IMG_PATH}2007_005844.jpg"
 BOTTLE="${IMG_PATH}2008_007811.jpg"
 BIRD="${IMG_PATH}2007_002094.jpg"
 TRAIN="${IMG_PATH}2007_000042.jpg"
 AIRPLANE="${IMG_PATH}2007_000033.jpg"
 
 GPU=2
 
 #NET="DeepLab-LargeFOV-Semi-Bbox-Fixed/deploy21.prototxt"
 #NET="DeepLab-LargeFOV/deploy21.prototxt"
NET="DeepLab-LargeFOV/deploy4.prototxt"
#NET="exper/voc12/config/DeepLab-LargeFOV/train.prototxt"
 
 #MODEL="exper/voc12/model/DeepLab-LargeFOV-Semi-Bbox-Fixed/train_iter_6000.caffemodel"
 #MODEL="exper/voc12/model/DeepLab-LargeFOV/train2_iter_12000.caffemodel"                                                                                                                                            
# MODEL="exper/voc12/model/DeepLab-LargeFOV_TMP/train_iter_6000.caffemodel"
MODEL="exper/voc12/model/DeepLab-LargeFOV/train2_iter_8000.caffemodel"
 
 #IMAGES="${CHAIR} ${BOTTLE} ${BIRD}"
 IMAGES="${BIRD}"
 
 BIRD2="${IMG_PATH}2010_004994.jpg"
 BOTTLE2="${IMG_PATH}2007_000346.jpg"
 CHAIR2="${IMG_PATH}2008_000673.jpg"
 IMAGES2="${CHAIR2} ${BOTTLE2} ${BIRD2} ${CHAIR} ${BOTTLE} ${BIRD} ${TRAIN} ${AIRPLANE}"

DOG="${IMG_PATH}_0djE279Srg_00015.jpg"
DOG2="${IMG_PATH}_0djE279Srg_00030.jpg"
DOG3="${IMG_PATH}_0djE279Srg_00045.jpg"

CAR="${IMG_PATH}_1MUXIam4lA_00015.jpg"
CAR2="${IMG_PATH}_1MUXIam4lA_00030.jpg"
CAR3="${IMG_PATH}_1MUXIam4lA_00045.jpg"

CAT="${IMG_PATH}_1UY2k5Mu3o_00030.jpg"
CAT2="${IMG_PATH}_1UY2k5Mu3o_00060.jpg"
CAT3="${IMG_PATH}_1UY2k5Mu3o_00090.jpg"
CAT4="${IMG_PATH}_1UY2k5Mu3o_00120.jpg"
CAT5="${IMG_PATH}_1UY2k5Mu3o_00150.jpg"

IMAGES3="${DOG} ${DOG2} ${DOG3} ${CAR} ${CAR2} ${CAR3} ${CAT} ${CAT2} ${CAT3} ${CAT4} ${CAT5}"
 
python deeplab.py ${GPU} ${NET} ${MODEL} ${IMAGES3}

