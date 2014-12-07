#!/bin/bash
dataDir="${PROJ_DIR}dl-image-enhance/data/uniform_set_100_v2_xpro"
cpDir=${dataDir}/convnet_checkpoints/ConvNet__2014-12-02_22.11.56zyan3-server2
cpName=2000.9

python imageEnhancer.py -f ${cpDir}/${cpName} --out-img-dir=${cpDir}_summary --label-layer=reglayer_Lab --gpu=1 --super-pixel-enhance=0 --num-proc=8 --fredo-image-processing=0 --do-enhance-image=1 --transform-vis=0  
#--test-image-id-file=${dataDir}/uniform_set_115_v2_xpro_test_id_addition.txt
