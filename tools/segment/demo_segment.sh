#!/bin/bash
PPM_DIR=/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/export_2/InputAsShotZeroed_ppm 
SIGMA=0.5
K=20
MIN=20
MAX=80

IMG="a0196-2004-01-25 13-34-10 CRW_3079"

./segment $SIGMA $K $MIN $MAX  "$PPM_DIR/$IMG.ppm" "$IMG.vis"
