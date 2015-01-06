#!/bin/bash
#prepare training batches for first effect (foreground highlighting) on uniform_set_100 dataset

inDataDir="../../dl-image-enhance/data/uniform_set"
enhDataDir="../../dl-image-enhance/data/uniform_set_watercolor"
train_image_id_file="uniform_set_train_id.txt"
test_image_id_file="uniform_set_test_id.txt"
data_save_dir="uniform_set_watercolor_3K_30_batch_seg_voting_k0.10"

python trainPatchPreparer.py --num-proc=20 --ori-img-folder=${inDataDir}/uniform_set_autotone_tif --img-seg-dir=${inDataDir}/uniform_set_autotone_seg_3k --sem-integral-map-dir=${inDataDir}/context/uniform_set_parts_seg_voting_k0.10_cleaned --semantic-map-dir=${inDataDir}/context/uniform_set_seg_voting_response_map_k0.10_cleaned --train-image-id-file=${inDataDir}/${train_image_id_file} --test-image-id-file=${inDataDir}/${test_image_id_file}  --enh-img-folder=${enhDataDir}/uniform_set_watercolor_tif  --data-save-path=${enhDataDir}/${data_save_dir} --num-batches=10 --preload-all-ori-img=1 --segment-random-sample-num=30 --do-compute-image-global-ftr=0 --do-local-context-SPM-regions=0 --do-compute-color-integral-map=0 --do-write-color-batch-files=1 --context-ftr-baseline=0
