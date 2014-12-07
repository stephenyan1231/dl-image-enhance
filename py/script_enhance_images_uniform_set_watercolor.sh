#!/bin/bash
dataDir="${PROJ_DIR}dl-image-enhance/data/uniform_set_watercolor"
cpDir=${dataDir}/convnet_checkpoints/ConvNet__2014-12-07_10.32.22zyan3-server2
cpName=6000.9

python imageEnhancer.py -f ${cpDir}/${cpName} --out-img-dir=${cpDir}_summary --label-layer=reglayer_Lab --gpu=0 --super-pixel-enhance=1 --num-proc=8 --do-enhance-image=1 --transform-vis=0 --super-pixel-enhance-uniform-color=1
