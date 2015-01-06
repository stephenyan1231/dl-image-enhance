#!/bin/bash
dataDir="../../dl-image-enhance/data/uniform_set_xpro"
cpDir=${dataDir}/convnet_checkpoints/ConvNet__2014-12-07_10.10.12zyan3-server2
cpName=2000.9

python imageEnhancer.py -f ${cpDir}/${cpName} --out-img-dir=${cpDir}_summary --label-layer=reglayer_Lab --gpu=0 --super-pixel-enhance=1 --num-proc=8 --do-enhance-image=1 --transform-vis=0