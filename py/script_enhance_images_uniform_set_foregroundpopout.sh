#!/bin/bash
dataDir="${PROJ_DIR}dl-image-enhance/data/uniform_set_foregroundpopout"
cpDir=${dataDir}/convnet_checkpoints/ConvNet__2014-12-06_22.23.55zyan3-server2
cpName=3000.9

python imageEnhancer.py -f ${cpDir}/${cpName} --out-img-dir=${cpDir}_summary --label-layer=reglayer_Lab --gpu=0 --super-pixel-enhance=1 --num-proc=8 --do-enhance-image=1 --transform-vis=0  