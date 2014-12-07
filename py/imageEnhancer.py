'''Copyright (c) 2014 Zhicheng Yan (zhicheng.yan@live.com)
'''
import sys
import os
sys.path.append(os.environ['PROJ_DIR'] + 'cuda-convnet-plus/cuda-convnet-plus/py')
import time
import fnmatch
import numpy as np
import scipy
import scipy.ndimage
import scipy.io
import scipy.misc
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
import multiprocessing as mtp
from skimage import color
from subprocess import call

from utilCnnImageEnhance import *
from util_image import * 
from convnet import *
from trainPatchPreparer import *

class ImageEnhancerError(Exception):
    pass

class ImageEnhancer(ConvNet):
    BATCH_NUM_IMG = 2048 * 2
    NUM_ROWS_PER_BATCH = 40
    
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)
        self.NUM_ROWS_PER_BATCH = 20
        
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', \
                              'test_batch_range', 'regress_L_channel_only', \
                              'use_local_context_ftr', 'layer_verbose'):
                op.delete_option(option)
                
        op.add_option("out-img-dir", "out_img_dir", StringOptionParser, \
                      "folder to store output (enhanced) images", default="")       
#         op.add_option("show-image-triplet", 'show_image_triplet', BooleanOptionParser, \
#                       "display image triplet (input,enhanced,groundtruth)", default=False)
        op.add_option("label-layer", 'label_layer', \
                      StringOptionParser, "name of label layer", default="")
        op.add_option("transform-vis", 'transform_vis', \
                    BooleanOptionParser, "visualize predicted transform", default=False)                                        
        op.add_option("super-pixel-enhance", 'super_pixel_enhance', BooleanOptionParser, \
                      "use super pixel as unit in enhancement stage", default=0)        
        op.add_option("super-pixel-enhance-post-process", 'super_pixel_enhance_post_process', BooleanOptionParser, \
                      "do post processing/smooth after superpixel-based enhancement", default=0, requires=['super_pixel_enhance'])                
        op.add_option("super-pixel-enhance-uniform-color", 'super_pixel_enhance_uniform_color', IntegerOptionParser, \
                      "assign an uniform color to superpixel in superpixel-based enhancement ", default=0, requires=['super-pixel-enhance'])        
        op.add_option("num-proc", "num_proc", IntegerOptionParser, "number of parallel processes", \
                      default=2)
        op.add_option("fredo-image-processing", 'fredo_image_processing', BooleanOptionParser,
                      "increase exposure by 1.5 and normalize L channel", default=0)
#         op.add_option("in-img-dir", 'in_img_dir', StringOptionParser,
#                       "overwrite input image folder in batches.meta", default='')        
#         op.add_option("enh-img-dir", 'enh_img_dir', StringOptionParser,
#                       "overwrite enhanced image folder in batches.meta", default='')         
        op.add_option("test-image-id-file", "test_image_id_file", StringOptionParser, \
                      "text file of test image id list", default="")  # allow to overwrite 'test_image_id_file' in meta file     
#         op.add_option("poisson-reconstruct-data-dir", "poisson_reconstruct_data_dir", \
#                       StringOptionParser, "folder to save txt data for Poisson reconstruction", default="")        
#         op.add_option("ori-img-edge-folder", "ori_img_edge_folder", StringOptionParser, \
#                       "folder of original image edge detection results", default="")
#         op.add_option("label-layer-gradmag", 'label_layer_gradmag', \
#                       StringOptionParser, "name of label layer representing gradient magnitude transform", \
#                       default="")                           
        op.add_option("checkpoints-dir", 'checkpoints_dir', \
                      StringOptionParser, "checkpoints folder", \
                      default="")          
#         op.add_option("ratio-on-edge", "ratio_on_edge", FloatOptionParser, \
#                       "ratio between color and gradient mag on edge pixels", \
#                       default=0.01)
#         op.add_option("ratio-off-edge", "ratio_off_edge", FloatOptionParser, \
#                       "ratio between color and gradient mag on edge pixels", \
#                       default=0.05)         
        op.add_option("do-enhance-image", "do_enhance_image", BooleanOptionParser, \
                      "do image enhancement?", default=False)               
#         op.add_option("do-poisson-reconstruct", "do_poisson_reconstruct", BooleanOptionParser, \
#                       "do poisson reconstruction?", default=False)
        op.add_option("color-regression", "color_regression", BooleanOptionParser, \
                      "regress color directly?", default=False)                
        
        op.options['load_file'].default = None
        return op
    
    def init_model_state(self):
        ConvNet.init_model_state(self)
        self.op.print_values()
        if self.op.get_value('label_layer'):
            self.label_layer_idx = self.get_layer_idx(self.op.get_value('label_layer'))
#         if self.op.get_value('label_layer_gradmag'):
#             self.label_layer_gradmag_idx = self.get_layer_idx(self.op.get_value('label_layer_gradmag'))
        self.tr_batches_meta = self.train_data_provider.get_batches_meta()          
        
        self.in_img_dir = self.tr_batches_meta['in_img_dir']
        self.img_seg_dir = self.tr_batches_meta['img_seg_dir']
        self.enh_img_dir = self.tr_batches_meta['enh_img_dir']
        self.semantic_map_dir = self.tr_batches_meta['semantic_map_dir']
        self.sem_integral_map_dir = self.tr_batches_meta['sem_integral_map_dir']
        if 'img_color_integral_dir' in self.tr_batches_meta.keys():
            self.img_color_integral_dir = self.tr_batches_meta['img_color_integral_dir']
        else:
            self.img_color_integral_dir = None
        if self.test_image_id_file:
            self.test_imgs = readImgNameFile(self.test_image_id_file)
        else:
            self.test_imgs = self.tr_batches_meta['ts_imgs']               
        
        if self.num_proc > 1:
            self.pool = mtp.Pool(processes=self.num_proc)
        else:
            self.pool = None
            
    def clear(self):
        if self.pool:
            self.pool.close()
    ''' testing stage. Use NN regressed color mapping to enhance input pixels '''
    def enhance_images(self):
        if not self.do_enhance_image:
            print 'skip and return'
            return
        self.op.print_values()
        if not os.path.exists(self.out_img_dir):
            os.mkdir(self.out_img_dir)
        
        if self.super_pixel_enhance:
            self.out_img_dir = os.path.join(self.out_img_dir,'superpixel')
        else:
            self.out_img_dir = os.path.join(self.out_img_dir,'pixel')
        if not os.path.exists(self.out_img_dir):
            os.mkdir(self.out_img_dir)
        
        img_size = self.tr_batches_meta['img_size']
        
        if self.use_local_context_ftr:
            local_context_paras = self.tr_batches_meta['local_context_paras']
            if self.use_local_context_color_ftr:
                local_context_color_paras = self.tr_batches_meta['local_context_color_paras']
        
        patch_size, num_colors = self.train_data_provider.get_img_size_num_colors()
        patch_hs = (patch_size - 1) / 2
        
        if not self.color_regression:
            basis_dim = self.train_data_provider.get_data_dims(self.train_data_provider.COLOR_BASIS_ID)
            num_regress_channel = self.train_data_provider.get_data_dims(self.train_data_provider.ENH_COLOR_ID)
            label_dim = basis_dim * num_regress_channel
        else:
            label_dim = 3
        
        num_in_imgs = len(self.test_imgs)
        L_dist, L_RMSE, Lab_dist, Lab_RMSE, num_pix = \
        np.zeros((num_in_imgs)), np.zeros((num_in_imgs)), np.zeros((num_in_imgs)), \
        np.zeros((num_in_imgs)), np.zeros((num_in_imgs))          
        
        if self.transform_vis:
            self.out_transform_vis_dir = os.path.join(self.out_img_dir, 'transform_vis')
            if not os.path.exists(self.out_transform_vis_dir):
                os.mkdir(self.out_transform_vis_dir)
        for i in range(num_in_imgs):
            st = time.time()
            print 'processing %d th out of %d images: %s' % (i, num_in_imgs, self.test_imgs[i])
            if self.fredo_image_processing == 1:
                in_img = read_tiff_16bit_img_into_LAB(os.path.join(self.in_img_dir, self.test_imgs[i] + '.tif'), 1.5, False)
                enh_img = read_tiff_16bit_img_into_LAB(os.path.join(self.enh_img_dir, self.test_imgs[i] + '.tif'), 0, False)                
            else:
                in_img = read_tiff_16bit_img_into_LAB(os.path.join(self.in_img_dir, self.test_imgs[i] + '.tif'))
                enh_img = read_tiff_16bit_img_into_LAB(os.path.join(self.enh_img_dir, self.test_imgs[i] + '.tif'))
            if self.super_pixel_enhance:
                in_img_seg = load_segment(os.path.join(self.img_seg_dir, self.test_imgs[i] + '.seg'), in_img.shape)
            
            assert in_img.shape == enh_img.shape
            h, w, ch = in_img.shape[0], in_img.shape[1], in_img.shape[2]
            assert ch == num_colors
            if not self.super_pixel_enhance:
                posx, posy = np.meshgrid(range(w), range(h))
                posx_flat, posy_flat = posx.flatten(), posy.flatten()
                num_patch = h * w
                num_batch = div_up(h, self.NUM_ROWS_PER_BATCH)
            else:
                num_seg = len(in_img_seg)
                print '%d segments' % num_seg
                posx_flat = np.zeros((num_seg), dtype=np.uint32)
                posy_flat = np.zeros((num_seg), dtype=np.uint32)
                for j in range(num_seg):
                    l_seg = np.array(in_img_seg[j])
                    l_seg_x, l_seg_y = l_seg % w, l_seg / w
                    posx_flat[j] = np.round(np.mean(l_seg_x))
                    posy_flat[j] = np.round(np.mean(l_seg_y))
                num_patch = num_seg
                num_batch = div_up(num_patch, self.BATCH_NUM_IMG)
                        
            predLb = np.zeros((num_patch, label_dim), dtype=np.single)
            
            expandInImg = get_expanded_img(in_img, patch_hs)          
        
            if self.use_local_context_ftr:
                contextSemIntegralMap = \
                scipy.io.loadmat(os.path.join(self.sem_integral_map_dir, self.test_imgs[i]+'.mat'))
                contextSemIntegralMap = contextSemIntegralMap['maps'].reshape((local_context_paras['label_num']))
                
                semMap = scipy.io.loadmat(os.path.join(self.semantic_map_dir, self.test_imgs[i]+'.mat'))
                semMap = semMap['responseMap']
                if self.use_local_context_color_ftr:
                    colorIntegralMap_ = scipy.io.loadmat(os.path.join(self.img_color_integral_dir, self.test_imgs[i]+'.mat'))
                    colorIntegralMap = colorIntegralMap_['colorIntegralMaps']
    
            pix_ftr_dim = CENTRAL_PX_FEATURE_DIM if CENTRAL_PX_FEATURE_DIM < 4 else next_4_multiple(CENTRAL_PX_FEATURE_DIM)
            
            ''' pixel-wise enhancement '''
            if not self.super_pixel_enhance:
                ''' memory allocation is slow. So pre-allocate memory. Reuse it for many times '''
                pix_ftr = np.zeros((self.NUM_ROWS_PER_BATCH * w, pix_ftr_dim), dtype=np.single)                
                if self.use_local_context_ftr:
                    contextSemFtr = np.zeros((self.NUM_ROWS_PER_BATCH * w, local_context_paras['ftr_dim']), dtype=np.single)        
                if self.use_local_context_color_ftr:
                    local_context_color_ftr = np.zeros\
                    ((self.NUM_ROWS_PER_BATCH * w, local_context_color_paras['hist_bin_num'] * 2), dtype=np.single)
                
                st = time.time()

                p = 0
                batch = 0
                                                      
                elapsed_time_data_prepare, elapsed_local_context, elapsed_dp, elapsed_dp1, elapsed_dp2 = \
                0, 0, 0, 0, 0
                
                start_row, end_row, data_ready, start_write = 0, 0, 0, 0
                while batch < num_batch:
                    if data_ready == 1 and batch < num_batch:
                        batch_size = next_batch_size
                        data = next_data
                        l_pred_labels = np.zeros((batch_size, label_dim), dtype=np.single)
                        self.libmodel.startFeatureWriter(data + [l_pred_labels], self.label_layer_idx)
                        batch += 1
                        start_write = 1
                    else:
                        start_write = 0
                    if batch < num_batch:
                        st_time2 = time.time()
                        start_row = end_row
                        end_row = min(end_row + self.NUM_ROWS_PER_BATCH, h)
                        next_batch_size = (end_row - start_row) * w
                        pix_ftr = pix_ftr[:next_batch_size,:]
                        
                        pix_x, pix_y = np.meshgrid(range(w), range(start_row, end_row))
                        pix_x, pix_y = pix_x.flatten(), pix_y.flatten()                        
                        get_pixel_feature_v2(in_img, pix_x, pix_y,pix_ftr)
                        
                        if self.use_local_context_ftr:
                            contextSemFtr = contextSemFtr[:next_batch_size, :]
                            st_time_lc = time.time()
                            getPixLocContextV2\
                            ([pix_x + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                              pix_y + LOCAL_CONTEXT_MAP_APPEND_WIDTH, contextSemIntegralMap, \
                              local_context_paras, self.pool, contextSemFtr, self.num_proc])
                            elapsed_local_context += time.time() - st_time_lc
                                
                        if self.use_local_context_color_ftr:
                            local_context_color_ftr = local_context_color_ftr[:next_batch_size, :]
                            l_p = 0
                            for row in range(start_row, end_row):
                                for col in range(w):
                                    get_local_context_color\
                                    (in_img[:, :, 1:3], col, row, local_context_color_paras, local_context_color_ftr[l_p, :])
                                    l_p = l_p + 1
                                                                  
                        gt_color = enh_img[start_row:end_row, :, :].reshape((next_batch_size, 3))
                        gt_color = gt_color.swapaxes(0, 1)                                             
                        st_time3 = time.time()
                                                
                        
                        if self.use_local_context_ftr:
                            context_ftr_t = contextSemFtr.transpose()
                            if self.use_local_context_color_ftr:
                                local_context_color_ftr_t = local_context_color_ftr.transpose()
                                context_mean_color_ftr_t = context_mean_color_ftr.transpose()
                                aux_data = [[self.test_imgs[i]] * next_batch_size, gt_color, context_ftr_t, \
                                            local_context_color_ftr_t, context_mean_color_ftr_t]
                            else:
                                aux_data = [[self.test_imgs[i]] * next_batch_size, gt_color, context_ftr_t]
                        else:
                            aux_data = [[self.test_imgs[i]] * next_batch_size, gt_color]
                        next_data_all = self.train_data_provider.prepare_batch_data\
                        (pix_ftr.transpose(), aux_data)                                       
                        next_data, elapsed_dp_12 = next_data_all[0], next_data_all[1]
                        
                        elapsed_dp += time.time() - st_time3
                        elapsed_dp1 += elapsed_dp_12[0]
                        elapsed_dp2 += elapsed_dp_12[1]
                        elapsed_time_data_prepare += time.time() - st_time2
                        
                        data_ready = 1
                    else:
                        data_ready = 0
                            
                    if  start_write == 1:
                        batch_output = self.finish_batch()
                        predLb[p:p + batch_size, :] = l_pred_labels
                        p += batch_size


                elapsed_time = time.time() - st
                
                if not self.color_regression:
                    ''' regress quadratic color basis transform '''
                    regress_ch = label_dim / basis_dim
                    predLb = predLb.reshape((h, w, regress_ch, basis_dim))
                    if self.transform_vis:
                        l_pred_labels = np.zeros((h, w, 3),dtype=np.single)
                        for j in range(basis_dim):
                            for k in range(regress_ch):
                                out_vis_path = os.path.join\
                                (self.out_transform_vis_dir, self.test_imgs[i] + '_vis_' + '%d_%d' % (k,j) + '.png')                                
                                color_transform = predLb[:, :, k, j]
                                l_pred_labels[:, :, :] = color_transform[:,:,np.newaxis]
                                l_min,l_max=np.min(l_pred_labels[:,:,0]),np.max(l_pred_labels[:,:,0])
                                l_pred_labels[:,:,:] = (l_pred_labels[:,:,:] - l_min) / (l_max - l_min)
                                scipy.misc.imsave(out_vis_path, l_pred_labels)
                    out_img = np.zeros((h, w, ch))
                           
                    predLb = predLb.reshape((h, w, label_dim))         
                    basis = quad_poly_color_basis(in_img.reshape((h * w, ch)))
                    assert label_dim % basis_dim == 0
                    basis = basis.reshape((h, w, basis_dim))
                    if self.regress_L_channel_only:
                        pred_L = np.sum(predLb * basis, axis=2)
                        out_img[:, :, 0] = pred_L
                        out_img[:, :, 1:3] = enh_img[:, :, 1:3]
                    else:
                        
                        out_img[:,:,:] = np.sum(predLb.reshape((h, w, regress_ch, basis_dim)) * basis[:, :, np.newaxis, :], \
                                            axis=3)
                else:
                    '''regress Lab color directly'''
                    out_img = predLb.reshape((h,w,label_dim))
            else:
                ''' super-pixel enhancement'''
                print 'super pixel-based enhancement'
                if self.use_local_context_ftr:
                    print '\tuse context semantic feature'
                    if self.use_local_context_color_ftr:
                        print '\tuse context color feature'
                pix_ftr = np.zeros((self.BATCH_NUM_IMG, pix_ftr_dim), dtype=np.single)
                if self.use_local_context_ftr:
                    contextSemFtr = np.zeros((self.BATCH_NUM_IMG, local_context_paras['ftr_dim']), dtype=np.single)
                    if self.use_local_context_color_ftr:
                        pix_color_hist_ftr = \
                        np.zeros((self.BATCH_NUM_IMG, local_context_color_paras['hist_bin_num']*2), dtype=np.single)
                
                st = time.time()
                p = 0
                batch = 0

                batch_size = 0
                start_pos, end_pos, data_ready, start_write = 0, 0, 0, 0
                elapsed_time_data_prepare, elapsed_dp, elapsed_dp1, elapsed_dp2 = 0, 0, 0, 0
                while batch < num_batch:
                    if data_ready == 1 and batch < num_batch:
                        batch_size = next_batch_size
                        data = next_data
                        l_pred_labels = np.zeros((batch_size, label_dim), dtype=np.single)
                        self.libmodel.startFeatureWriter(data + [l_pred_labels], self.label_layer_idx)
                        batch += 1
                        start_write = 1
                    else:
                        start_write = 0
                    if batch < num_batch:
                        st_time2 = time.time()
                        start_pos = end_pos
                        end_pos = min(num_patch, start_pos + self.BATCH_NUM_IMG)
                        next_batch_size = end_pos - start_pos
                        l_posx, l_posy = posx_flat[start_pos:end_pos], posy_flat[start_pos:end_pos]

                        pix_ftr = pix_ftr[:next_batch_size, :]
                        get_pixel_feature_v2\
                        (in_img, l_posx, l_posy, pix_ftr[:, :CENTRAL_PX_FEATURE_DIM])                        
                        
                        gt_color = enh_img[l_posy, l_posx, :].reshape((next_batch_size, 3))
                        gt_color = gt_color.swapaxes(0, 1)
                            
                        st_time3 = time.time()
                        if self.use_local_context_ftr:
                            contextSemFtr = contextSemFtr[:next_batch_size, :]
                            getPixLocContextV2([l_posx + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                                                        l_posy + LOCAL_CONTEXT_MAP_APPEND_WIDTH, contextSemIntegralMap, \
                                                        local_context_paras, self.pool, contextSemFtr, self.num_proc])
                            
                            if self.use_local_context_color_ftr:                                
                                pix_color_hist_ftr = pix_color_hist_ftr[:next_batch_size,:]
                                for j in range(next_batch_size):
                                    pix_color_hist_ftr[j,:]=get_local_context_color\
                                    (in_img[:,:,1:3], l_posx[j],l_posy[j],local_context_color_paras)
                                contextColorFtr = getPixContextColorFtr\
                                (l_posx+LOCAL_CONTEXT_MAP_APPEND_WIDTH, l_posy+LOCAL_CONTEXT_MAP_APPEND_WIDTH,\
                                 colorIntegralMap, self.tr_batches_meta['local_context_paras'])
                                
                                next_data_all = self.train_data_provider.prepare_batch_data\
                                (pix_ftr.transpose(), [[self.test_imgs[i]] * next_batch_size, gt_color, \
                                           contextSemFtr.transpose(), pix_color_hist_ftr.transpose(), \
                                           contextColorFtr.transpose()])                          
                            else:
                                next_data_all = self.train_data_provider.prepare_batch_data\
                                (pix_ftr.transpose(), [[self.test_imgs[i]] * next_batch_size, gt_color, \
                                                       contextSemFtr.transpose()])                            
                        else:
                            next_data_all = self.train_data_provider.prepare_batch_data\
                            (pix_ftr.transpose(), [[self.test_imgs[i]] * next_batch_size, gt_color])
                        next_data, elapsed_dp_12 = next_data_all[0], next_data_all[1]
                        
                        elapsed_dp += time.time() - st_time3
                        elapsed_dp1 += elapsed_dp_12[0]
                        elapsed_dp2 += elapsed_dp_12[1]
                        elapsed_time_data_prepare += time.time() - st_time2
                        
                        data_ready = 1
                    else:
                        data_ready = 0
                    if start_write == 1:
                        batch_output = self.finish_batch()
                        predLb[p:p + batch_size, :] = l_pred_labels
                        p += batch_size
                
                basis = quad_poly_color_basis(in_img.reshape((h * w, ch)))
                basis_dim = basis.shape[1]
                assert label_dim % basis_dim == 0
                regress_ch = label_dim / basis_dim
                basis = basis.reshape((h * w, basis_dim))
                                
                enh_img = enh_img.reshape((h * w, ch))
                
                if self.transform_vis:
                    transform = np.zeros((label_dim, h * w))
                    for j in range(num_seg):
                        seg = np.array(in_img_seg[j])
                        transform[:, seg] = predLb[j, :,np.newaxis]
                    for j in range(label_dim):
                        out_vis_path = os.path.join(self.out_transform_vis_dir, self.test_imgs[i] + '_vis_' + '%d' % j + '.png')
                        out_vis_img = transform[j, :]
                        out_vis_img = (out_vis_img - np.min(out_vis_img)) / (np.max(out_vis_img) - np.min(out_vis_img))
                        scipy.misc.imsave(out_vis_path, out_vis_img.reshape((h, w)))
                        
                st_time = time.time()
                if not self.super_pixel_enhance_post_process:
                    ''' no post smoothing operation '''
                    out_img = np.zeros((h * w, ch))
                    for j in range(num_seg):
                        seg = np.array(in_img_seg[j])
                        if self.regress_L_channel_only:
                            out_img[seg, 0] = np.sum(basis[seg, :] * predLb[j, :], axis=1)
                        else:
                            l_pred_colors = np.dot(predLb[j, :].reshape((regress_ch, basis_dim)), basis[seg, :].transpose())
                            if self.super_pixel_enhance_uniform_color:
                                if self.super_pixel_enhance_uniform_color == 1:
                                    out_img[seg, :] = np.mean(l_pred_colors,axis=1)
                                else:
                                    out_img[seg, :] = np.mean(l_pred_colors, axis=1)
                            else:
                                out_img[seg, :] = l_pred_colors.transpose()
                    if self.regress_L_channel_only:
                        out_img[:, 1:3] = enh_img[:, 1:3]
                    out_img = out_img.reshape((h, w, ch))

                else:
                    ''' do post-processing to smooth superpixel-based enhanced output image
                    assign weights to different segments based on distance between segment mean color and pixel color '''                    
                    print 'do post-processing for super pixel enhancement'
                    out_img = np.zeros((h * w, ch))
                    dilate_width = 1  # dilate outwards by a constant number of pixels
                    max_num_seg = 10  # the maximum nmber of associated segments with each pixel. high value leads to better results but slow speed
                    color_sigma = 20.0
                    
                    pix_seg_id = np.zeros((h * w, max_num_seg), dtype=np.int32)
                    pix_seg_id[:, :] = num_seg
                    pix_seg_weights = np.zeros((h * w, max_num_seg))
                    pix_seg_count = np.zeros((h * w), dtype=np.int8)
                    
                    st_time_1 = time.time()
                    affected_ids = get_affected_pixels(self.pool, h, w, in_img_seg, dilate_width)
                    elapsed_extend_seg = (time.time() - st_time_1)
                    
                    included_mask = np.zeros((h * w), dtype=np.bool)
                
                    for j in range(num_seg):
                        l_seg = np.array(in_img_seg[j])
                        # dilate by a constant number of pixels
                        l_seg_x, l_seg_y = l_seg % w, l_seg / w
#                         l_seg_cx, l_seg_cy = np.mean(l_seg_x), np.mean(l_seg_y)
                        l_seg_color = in_img[l_seg_y, l_seg_x, :]
                        l_seg_color = np.mean(l_seg_color, axis=0)
                        
                        l_included_id = affected_ids[j]
                        l_included_x, l_included_y = l_included_id % w, l_included_id / w
                        
                        included_mask[l_included_id] = 1

                        l_included_color = in_img[l_included_y, l_included_x, :]
                        l_weights = np.exp(-np.sum((l_included_color - l_seg_color[np.newaxis, :]) ** 2, axis=1) / (color_sigma ** 2))
                        
                        ids = np.nonzero(pix_seg_count[l_included_id] < max_num_seg)
                        pix_seg_id[l_included_id[ids[0]], pix_seg_count[l_included_id[ids[0]]]] = j
                        pix_seg_weights[l_included_id[ids[0]], pix_seg_count[l_included_id[ids[0]]]] = l_weights[ids[0]]
                        pix_seg_count[l_included_id[ids[0]]] += 1
                    
                    ids = np.nonzero(included_mask == 0)
                    if len(ids[0]) > 0:
                        print 'not included ids', ids[0]
                    
                    ''' normalize weights for each pixel '''
                    pix_seg_weight_sum = np.sum(pix_seg_weights, axis=1)
                    ids = np.nonzero(pix_seg_weight_sum == 0)
                    if len(ids[0]) > 0:
                        print 'ids', ids[0]
                        print 'find zero weight sum'
                        print 'pix_seg_weights zero sum', pix_seg_weights[ids[0], :]
                        sys, exit()
                    pix_seg_weight_normed = pix_seg_weights / pix_seg_weight_sum[:, np.newaxis]
                                    
                    st_time_2 = time.time()
                    ''' append one dummy zero-valued row in the bottom '''
                    predLb = np.vstack((predLb, np.zeros((1, label_dim))))  
                    ''' compute color mapping transform for each pixel '''
                    pix_seg_id = pix_seg_id.reshape((h * w * max_num_seg))
                    pix_seg_weight_normed = pix_seg_weight_normed.reshape((h * w * max_num_seg))
                    smooth_transform = predLb[pix_seg_id, :] * pix_seg_weight_normed[:, np.newaxis]
                    smooth_transform = smooth_transform.reshape((h * w, max_num_seg, label_dim))
                    smooth_transform = np.sum(smooth_transform, axis=1)
                    elapsed_transform = time.time() - st_time_2
                    
                    if self.regress_L_channel_only:
                        out_img[:, 0] = np.sum(smooth_transform * basis, axis=1)
                        out_img[:, 1:3] = enh_img[:, 1:3]
                    else:
                        smooth_transform = smooth_transform.reshape((h * w, regress_ch, basis_dim))
                        out_img = np.sum(smooth_transform * basis[:, np.newaxis, :], axis=2)
                    out_img = out_img.reshape((h, w, ch))   
                    elapsed = time.time() - st_time
                    print 'post-processing time cost: %4.2f elapsed_extend_seg:%4.2f elapsed_transform:%4.2f' % \
                    (elapsed, elapsed_extend_seg, elapsed_transform)                 
                elapsed_time = time.time() - st_time
                print 'super-pixel-based enhancement elapsed_time:%f' % elapsed_time
                enh_img = enh_img.reshape((h, w, ch))
            out_img_path = os.path.join(self.out_img_dir, self.test_imgs[i] + ".png")            
            save_image_LAB_into_sRGB(out_img_path, out_img)
            
            elapsed_time = time.time() - st
            print 'predicting enhancement elapsed time:%f elapsed_time_data_prepare %f, elapsed_dp:%f'\
            % (elapsed_time, elapsed_time_data_prepare, elapsed_dp)            
            
            num_pix[i] = h * w
            
            L_dist[i] = np.sum(np.abs(out_img[:, :, 0] - enh_img[:, :, 0])) / num_pix[i]
            L_RMSE[i] = np.sqrt(np.sum((out_img[:, :, 0] - enh_img[:, :, 0]) ** 2) / num_pix[i])
            Lab_dist[i] = np.sum(np.sqrt(np.sum((out_img - enh_img) ** 2, axis=2))) / num_pix[i]
            Lab_RMSE[i] = np.sqrt(np.sum(((out_img - enh_img) ** 2)) / num_pix[i])
            
            print 'image %s L_dist %f L_RMSE %f Lab_dist:%f Lab_RMSE %f'\
            % (self.test_imgs[i], L_dist[i], L_RMSE[i], Lab_dist[i], Lab_RMSE[i])             
            print 'so far, mean L_dist %f L_RMSE %f Lab_dist:%f Lab_RMSE %f'\
            % (np.sum(L_dist) / (i + 1), np.sum(L_RMSE) / (i + 1), np.sum(Lab_dist) / (i + 1), np.sum(Lab_RMSE) / (i + 1))
            
            
            print '----------------------------'
        self.op.print_values()
        
        img_error_report_fid = open(os.path.join(self.out_img_dir, 'img_errors.txt'),'w')
        for i in range(len(self.test_imgs)):
            img_error_report_fid.write('%d th image: %s Lab dist:%f\n' % (i+1, self.test_imgs[i],Lab_dist[i]))
        img_error_report_fid.write('\n')
        img_error_report_fid.write('mean L2 distance in L channel:%f in Lab channels:%f' % (np.mean(L_dist), np.mean(Lab_dist)))
        img_error_report_fid.close()
        print 'mean L2 distance in L channel:%f in Lab channels:%f' % (np.mean(L_dist), np.mean(Lab_dist))
        
        res_dict = {}
        res_dict['test_imgs'] = self.test_imgs
        res_dict['L_RMSE'] = L_RMSE
        res_dict['Lab_RMSE'] = Lab_RMSE
        res_dict['L_dist'] = L_dist
        res_dict['Lab_dist'] = Lab_dist
        res_dict['num_pix'] = num_pix
        imgToLabDist = {}
        for i in range(num_in_imgs):
            imgToLabDist[self.test_imgs[i]] = Lab_dist[i]
        res_dict['imgToLabDist'] = imgToLabDist
        out_res_path = os.path.join(self.out_img_dir, 'enhance_summary')
        pickle(out_res_path, res_dict)
        

    def writeEnhancedLabimgToTxt(self, img_Lab, txt_dir):
        h, w, c = img_Lab.shape[0], img_Lab.shape[1], img_Lab.shape[2]
        img_Lab = img_Lab.reshape((h * w, c))
        txt_file_path = os.path.join(txt_dir, 'us.txt')
        f = open(txt_file_path, 'w')
        f.write('%d\n' % (h * w))
        f.write('%d\n' % w)
        for ch in range(3):
            for p in range(h * w):
                f.write('%f\n' % img_Lab[p, ch])
        f.close()
    
    def writeImgBinaryEdgeMaskToTxt(self, edge_pix_x, edge_pix_y, h, w, txt_dir):        
        txt_file_path = os.path.join(txt_dir, 'new_IsEdge.txt')
        binary_mask = np.zeros((h, w), dtype=np.uint32)
        binary_mask[edge_pix_y, edge_pix_x] = 1
        f = open(txt_file_path, 'w')
        f.write('%d\n' % (h * w))
        f.write('%d\n' % w)
        for i in range(h):
            for j in range(w):
                f.write('%d\n' % binary_mask[i, j])
        f.close()
    
    
    def write_divergence_to_txt(self, grad_mag, grad_angle, txt_dir):
        assert grad_mag.shape == grad_angle.shape
        h, w, c = grad_mag.shape[0], grad_mag.shape[1], grad_mag.shape[2]
        div = compute_divergence_from_gradient(grad_mag, grad_angle)
        txt_file_path = os.path.join(txt_dir, 'divs.txt')
        f = open(txt_file_path, 'w')
        f.write('%d\n' % (h * w))
        f.write('%d\n' % w)
        for ch in range(c):
            for i in range(h):
                for j in range(w):
                    f.write('%f\n' % div[i, j, ch])
        f.close()
    
    def do_Poisson_reconstruct(self):
        if not self.do_poisson_reconstruct:
            print 'skip and return'
            return
        
        print 'do_Poisson_reconstruct'
        cur_dir = os.getcwd()
        
        if not os.path.exists(self.poisson_reconstruct_data_dir):
            os.mkdir(self.poisson_reconstruct_data_dir)
        
#         if self.use_local_context_ftr:
        local_context_paras = self.tr_batches_meta['local_context_paras']
        
        num_in_imgs = len(self.test_imgs)
        gradmag_transform_dim = 2

        gradmag_L2_dist = np.zeros((num_in_imgs))
        
        new_img_save_dir = \
        os.path.join(self.poisson_reconstruct_data_dir, '%f_%f' % (self.ratio_on_edge, self.ratio_off_edge))
        if not os.path.exists(new_img_save_dir):
            os.mkdir(new_img_save_dir)
        
        for i in range(num_in_imgs):
            print 'processing %d th out of %d images: %s' % (i, num_in_imgs, self.test_imgs[i])            
            out_img_data_dir = os.path.join\
            (self.poisson_reconstruct_data_dir, self.test_imgs[i][:-4])
            if not os.path.exists(out_img_data_dir):
                os.mkdir(out_img_data_dir)

            os.chdir(out_img_data_dir)
            
            if self.fredo_image_processing == 1:
                in_img = read_tiff_16bit_img_into_LAB\
                (os.path.join(self.in_img_dir, self.test_imgs[i]), 1.5, False)
                gt_enh_img = read_tiff_16bit_img_into_LAB\
                (os.path.join(self.enh_img_dir, self.test_imgs[i]), 0, False)                
            else:
                in_img = read_tiff_16bit_img_into_LAB\
                (os.path.join(self.in_img_dir, self.test_imgs[i]))
                gt_enh_img = read_tiff_16bit_img_into_LAB\
                (os.path.join(self.enh_img_dir, self.test_imgs[i]))            
            assert in_img.shape == gt_enh_img.shape
            h, w, ch = in_img.shape[0], in_img.shape[1], in_img.shape[2]
            
            in_img_L = in_img[:, :, 0]
            in_img_grady_L, in_img_gradx_L = np.gradient(in_img_L)
            in_img_gradmag_L = np.sqrt(in_img_grady_L ** 2 + in_img_gradx_L ** 2)
            gt_enh_img_L = gt_enh_img[:, :, 0]
            gt_enh_img_grady_L, gt_enh_img_gradx_L = np.gradient(gt_enh_img_L)
            gt_enh_img_gradmag_L = np.sqrt(gt_enh_img_grady_L ** 2 + gt_enh_img_gradx_L ** 2)
            
            pred_enh_img_path = os.path.join(self.out_img_dir, self.test_imgs[i][:-4] + '.png')
            pred_enh_im_sRGB = scipy.misc.imread(pred_enh_img_path)
            pred_enh_im_Lab = color.rgb2lab(pred_enh_im_sRGB)
            
            # normalize L to [0,1],normalize a,b channels to [-1,1]
            normalizer = np.array([1.0 / 100, 1.0 / 128.0, 1.0 / 128.0])
            in_img_normed = in_img * normalizer[np.newaxis, np.newaxis, :]
            pred_enh_im_Lab_normed = pred_enh_im_Lab * normalizer[np.newaxis, np.newaxis, :]                        
            
            img_edge_mat_file_path = \
            os.path.join(self.ori_img_edge_folder, self.test_imgs[i][:-4] + '_edge.mat')            
            edge_pix = scipy.io.loadmat(img_edge_mat_file_path)
            edge_pix_y, edge_pix_x = np.int32(edge_pix['edge_pix_y'] - 1), \
            np.int32(edge_pix['edge_pix_x'] - 1)  # note, matlab uses 1-based index             
            edge_pix_y = edge_pix_y.reshape((edge_pix_y.shape[0]))
            edge_pix_x = edge_pix_x.reshape((edge_pix_x.shape[0]))
            assert edge_pix_y.shape == edge_pix_x.shape
            edge_pix_x, edge_pix_y = \
            get_extended_edge_pixel(edge_pix_x, edge_pix_y, h, w, extend_width=1)                        
            gt_enh_edge_pix_gradmag_L = gt_enh_img_gradmag_L[edge_pix_y, edge_pix_x]

            # prepare data for neural network
            batch_size = edge_pix_x.shape[0]
            print 'batch_size', batch_size
            edge_pix_L = in_img_L[edge_pix_y, edge_pix_x]
            edge_pix_gradmag_L = in_img_gradmag_L[edge_pix_y, edge_pix_x]
            if self.in_img_precomputed_context_ftr_dir:
                precomputed_context_ftr_path = os.path.join\
                (self.in_img_precomputed_context_ftr_dir, self.test_imgs[i][:-4] + '_context_ftr.mat')
                precomputed_context_ftr = scipy.io.loadmat(precomputed_context_ftr_path)
                precomputed_context_ftr = precomputed_context_ftr['context_ftr']                
                edge_pix_context_ftr = precomputed_context_ftr[edge_pix_y, edge_pix_x, :]
            else:
                '''compute context ftr on the fly'''
                context_map = \
                scipy.io.loadmat(os.path.join(self.sem_integral_map_dir, self.semIntegralMapFiles[i]))
                context_map = context_map['maps'].reshape((local_context_paras['label_num']))
                
                edge_pix_context_ftr = np.zeros\
                ((edge_pix_x.shape[0], local_context_paras['ftr_dim']), dtype=np.single)
                getPixLocContextV2\
                ([edge_pix_x + LOCAL_CONTEXT_MAP_APPEND_WIDTH,
                  edge_pix_y + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                  context_map, local_context_paras, self.pool, edge_pix_context_ftr, self.num_proc])

            data = self.train_data_provider.prepare_batch_data\
            (edge_pix_L.reshape((1, batch_size)), \
             [[self.in_img_id[i]] * batch_size, edge_pix_gradmag_L.reshape((1, batch_size)), edge_pix_context_ftr.transpose()])
            gradmag_transform = np.zeros((batch_size, gradmag_transform_dim), dtype=np.single)
            print 'start feature writer'
            st_time = time.time()
            self.libmodel.startFeatureWriter(data + [gradmag_transform], self.label_layer_gradmag_idx)
            self.finish_batch()
            elapsed = time.time() - st_time
            print 'startFeatureWriter elapsed time:%f' % elapsed
            
            pred_edge_pix_gradmag_L = np.exp(gradmag_transform[:, 0] * np.log(edge_pix_L) + gradmag_transform[:, 1])\
            *edge_pix_gradmag_L
            np.max(pred_edge_pix_gradmag_L), np.mean(pred_edge_pix_gradmag_L)
            gradmag_L2_dist[i] = np.mean(np.abs(pred_edge_pix_gradmag_L - gt_enh_edge_pix_gradmag_L))
            print 'prediction mean L2 distance between predicted and groundtruth grad-mag on edge pixels', \
            gradmag_L2_dist[i]
            # load predicted enhanced color image. write it to txt
            self.writeEnhancedLabimgToTxt(pred_enh_im_Lab_normed, out_img_data_dir)
            # write edge pixel mask to txt
            self.writeImgBinaryEdgeMaskToTxt(edge_pix_x, edge_pix_y, h, w, out_img_data_dir)

            gaus_smooth_sigma = 1.5
            grad_xy = get_color_gradient(in_img_normed, central_diff=True)
            grady_L, gradx_L = grad_xy[:, :, 3], grad_xy[:, :, 0]
            grady_a, gradx_a = grad_xy[:, :, 4], grad_xy[:, :, 1]
            grady_b, gradx_b = grad_xy[:, :, 5], grad_xy[:, :, 2]            

            grad_mag_L, grad_angle_L = compute_grad_mag_angle(gradx_L, grady_L)
            grad_mag_a, grad_angle_a = compute_grad_mag_angle(gradx_a, grady_a)
            grad_mag_b, grad_angle_b = compute_grad_mag_angle(gradx_b, grady_b)
            
            pred_edge_pix_gradmag_L_normed = pred_edge_pix_gradmag_L / 100.0
            pred_enh_im_edge_pix_gradmag_L = grad_mag_L[edge_pix_y, edge_pix_x]
            diff = np.mean(np.abs(pred_edge_pix_gradmag_L_normed - pred_enh_im_edge_pix_gradmag_L))
            print 'mean normalized L2 distance between predicted and already enhanced grad-mag', diff
            ''' compute divergence and write it to txt file '''
            grad_mag_target = np.zeros((h, w, ch), dtype=np.single)
            grad_angle_target = np.zeros((h, w, ch), dtype=np.single)
            
            grad_mag_target[:, :, 0] = grad_mag_L[:, :]
            grad_angle_target[:, :, 0] = grad_angle_L                
            grad_mag_target[:, :, 1] = grad_mag_a[:, :]
            grad_angle_target[:, :, 1] = grad_angle_a[:, :]
            grad_mag_target[:, :, 2] = grad_mag_b[:, :]
            grad_angle_target[:, :, 2] = grad_angle_b[:, :]
            
#             grad_mag_target[edge_pix_y, edge_pix_x, 0] = 0.9            
            grad_mag_target[edge_pix_y, edge_pix_x, 0] = pred_edge_pix_gradmag_L_normed            
            self.write_divergence_to_txt(grad_mag_target, grad_angle_target, out_img_data_dir)
            ''' copy 'PoissonRestruction.ext' into current folder '''
            command = \
            ['copy', os.path.join(self.checkpoints_dir, 'PoissonRestruction.exe'), "\"" + out_img_data_dir + "\""]
            command = " ".join(command)
            print 'command', command
            call(command, shell=True)
            print 'copying is completed'
            new_enhanced_img_txt_nm = 'newLab.txt'
            command = \
            ['PoissonRestruction.exe', new_enhanced_img_txt_nm, '%f' % self.ratio_on_edge, '%f' % self.ratio_off_edge]
            command = " ".join(command)
            print 'command: %s' % command
            call(command, shell=True)
            command = ['del', 'PoissonRestruction.exe']
            command = " ".join(command)
            call(command, shell=True)
            
            new_img_Lab = read_Lab_txt_into_Lab_img(h, w, new_enhanced_img_txt_nm)
            new_img_sRGB = color.lab2rgb(new_img_Lab)
            new_img_sRGB = clamp_sRGB_img(new_img_sRGB)
            # append outmost 1-pixel width region
            new_img_sRGB[0, 1:-1, :] = new_img_sRGB[1, 1:-1, :]
            new_img_sRGB[h - 1, 1:-1, :] = new_img_sRGB[h - 2, 1:-1, :]
            new_img_sRGB[1:-1, 0, :] = new_img_sRGB[1:-1, 1, :]
            new_img_sRGB[1:-1, w - 1, :] = new_img_sRGB[1:-1, w - 2, :]
            
            new_img_sRGB[0, 0, :] = new_img_sRGB[0, 1, :]
            new_img_sRGB[0, w - 1, :] = new_img_sRGB[0, w - 2, :]
            new_img_sRGB[h - 1, 0, :] = new_img_sRGB[h - 1, 1, :]
            new_img_sRGB[h - 1, w - 1, :] = new_img_sRGB[h - 1, w - 2, :]
            
            scipy.misc.imsave\
            (os.path.join(new_img_save_dir, self.test_imgs[i][:-4] + '.png'), new_img_sRGB)
            print '--------------------------------------------------------'
                       
                        
        summary_file_path = os.path.join(new_img_save_dir, 'summary')
        summ = {}
        summ['gradmag_L2_dist'] = gradmag_L2_dist
        summ['ratio_on_edge'] = self.ratio_on_edge
        summ['ratio_off_edge'] = self.ratio_off_edge
        pickle(summary_file_path, summ)
        
        os.chdir(cur_dir)

if __name__ == "__main__":
    op = ImageEnhancer.get_options_parser()
    op, load_dic = IGPUModel.parse_options(op)
    model = ImageEnhancer(op, load_dic)
    model.enhance_images()
    # model.do_Poisson_reconstruct() 
    model.clear()
