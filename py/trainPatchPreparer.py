'''Copyright (c) 2014 Zhicheng Yan (zhicheng.yan@live.com)
'''
import sys
import os
sys.path.append(os.environ['PROJ_DIR'] + 'cuda_convnet_plus/py')

from options import *
from util import *
from util_image import *
from utilCnnImageEnhance import *
from PCA import *

import os
import scipy
import scipy.ndimage
import scipy.stats
import scipy.io
import scipy.misc
import numpy as np
import fnmatch
from numpy import random as rd 
import multiprocessing as mtp
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
import time
import re


class PatchPreparerError(Exception):
    pass
    
class TrainPatchPreparer:
    def __init__(self, op):
        self.op = op

        for o in op.get_options_list():
            setattr(self, o.name, o.value)        
        op.print_values()
        self.init_local_context_ftr_paras()
        
        self.pool = mtp.Pool(processes=self.num_proc)
    
    def clear(self):
        if self.pool:
            self.pool.close()

    ''' initialize parameter setting for computing context feature
    '''
    def init_local_context_ftr_paras(self):
        context = {}
        context['label_num'] = SEMANTIC_LABELS_NUM
        context['radius'] = 3
        if not self.context_ftr_baseline:
            context['partition_num'] = 8
            context['layer_num'] = 4
             
            n_pooling_region = (context['layer_num'] - 1) * context['partition_num'] + 1
            if self.do_local_context_SPM_regions:
                n_pooling_region += (context['layer_num'] - 1)
             
            offsets = np.zeros((4, n_pooling_region), dtype=np.int32)
            area_pix_num = np.zeros((n_pooling_region), dtype=np.int32)
            for i in range(context['layer_num']):
                r = np.power(context['radius'], i + 1)
                if i == 0:
                    offsets[:, i] = [-r, -r, r, r]
                    area_pix_num[i] = (2 * r + 1) ** 2
                else:
                    rp = r / context['radius']
                    offsets[:, (i - 1) * 8 + 1] = [-r, -r, -rp - 1, -rp - 1]
                    offsets[:, (i - 1) * 8 + 2] = [-r, -rp, -rp - 1, rp]
                    offsets[:, (i - 1) * 8 + 3] = [-r, rp + 1, -rp - 1, r]
                    offsets[:, (i - 1) * 8 + 4] = [-rp, -r, rp, -rp - 1]
                    offsets[:, (i - 1) * 8 + 5] = [-rp, rp + 1, rp, r]
                    offsets[:, (i - 1) * 8 + 6] = [rp + 1, -r, r, -rp - 1]
                    offsets[:, (i - 1) * 8 + 7] = [rp + 1, -rp, r, rp]
                    offsets[:, (i - 1) * 8 + 8] = [rp + 1, rp + 1, r, r]
                     
                    area_pix_num[(i - 1) * 8 + 1] = (r - rp) * (r - rp)
                    area_pix_num[(i - 1) * 8 + 2] = (r - rp) * (2 * rp + 1)
                    area_pix_num[(i - 1) * 8 + 3] = (r - rp) * (r - rp)
                    area_pix_num[(i - 1) * 8 + 4] = (r - rp) * (2 * rp + 1)
                    area_pix_num[(i - 1) * 8 + 5] = (r - rp) * (2 * rp + 1)
                    area_pix_num[(i - 1) * 8 + 6] = (r - rp) * (r - rp)
                    area_pix_num[(i - 1) * 8 + 7] = (r - rp) * (2 * rp + 1)
                    area_pix_num[(i - 1) * 8 + 8] = (r - rp) * (r - rp)
            if self.do_local_context_SPM_regions:
                for i in range(context['layer_num'] - 1):
                    r = np.power(context['radius'], i + 2)
                    offsets[:, n_pooling_region - context['layer_num'] + 1 + i] = [-r, -r, r, r]
                    area_pix_num[n_pooling_region - context['layer_num'] + 1 + i] = (2 * r + 1) ** 2
        else:
            context['label_num'] = SEMANTIC_LABELS_NUM
            context['layer_num'] = 4
            
            n_pooling_region = 1             
            offsets = np.zeros((4, n_pooling_region), dtype=np.int32)
            area_pix_num = np.zeros((n_pooling_region), dtype=np.int32)
            
            r = np.power(context['radius'], context['layer_num'])
            offsets[:, 0] = [-r, -r, r, r]
            area_pix_num[0] = (2 * r + 1) ** 2
     
        context['offsets'] = offsets
        context['area_pix_num'] = area_pix_num
        context['ftr_dim'] = offsets.shape[1] * context['label_num']
        context['pool_region_num'] = offsets.shape[1]        
        
        if not self.img_color_integral_dir == None:
            self.local_context_color_paras = {}
            self.local_context_color_paras['hist_bin_num'] = 32
            self.local_context_color_paras['half_size'] = 20
        
        self.local_context_paras = context

    @staticmethod
    def parse_options(op):
        op.parse()
        op.eval_expr_defaults()
        return op
    
    
    def find_semantic_maps(self):
        if not self.semantic_map_dir:
            return
        print 'find_semantic_maps'
        for img in self.imgs:
            sem_f_p =os.path.join(self.semantic_map_dir,img+'.mat')
            if not os.path.exists(sem_f_p):
                raise PatchPreparerError('semantic map for %s is missing'%img)
        
    ''' find input and enhanced image pair.
        find train/test image ids'''
    def find_images(self):
        self.tr_imgs = readImgNameFile(self.train_image_id_file)
        self.ts_imgs = readImgNameFile(self.test_image_id_file)
        self.imgs = self.tr_imgs + self.ts_imgs
        
        for img in self.imgs:
            in_img_p = os.path.join(self.ori_img_folder,img+'.tif')
            enh_img_p = os.path.join(self.enh_img_folder,img+'.tif')
            if not os.path.exists(in_img_p):
                raise PatchPreparerError('input image %s is missing' % img)
            if not os.path.exists(enh_img_p):
                raise PatchPreparerError('enhanced image %s is missing' % img)                     
       
        print '%d,%d training/test images' % (len(self.tr_imgs), len(self.ts_imgs))
         
    ''' find precomputed segmentation files '''
    def find_segments(self):
        if self.img_seg_dir:
            for img in self.imgs:
                in_seg_p = os.path.join(self.img_seg_dir, img+'.seg')
                if not os.path.exists(in_seg_p):
                    raise PatchPreparerError('segmentation file for %s is missing' % img)

    ''' find precomputed semantic integral maps'''
    def find_semantic_integral_files(self):
        if self.sem_integral_map_dir:
            for img in self.imgs:
                sem_int_f_p=os.path.join(self.sem_integral_map_dir,img+'.mat')
                if not os.path.exists(sem_int_f_p):
                    raise PatchPreparerError('semantic integral map for %s is missing'%img)

    ''' find precomputed color integral maps '''
    def find_color_integral_files(self):
        if self.img_color_integral_dir:
            for img in self.imgs:
                color_int_f_p = os.path.join(self.img_color_integral_dir, img + '.mat')
                if not os.path.join(color_int_f_p):
                    raise PatchPreparerError('color integral map file for %s is missing' % img)

    ''' compute mean L1 norm 
        ftr[i][j]: i->image j->dimension'''
    def mean_L1(self, ftr):
        num = len(ftr)
        mean_l1 = np.sum(np.abs(ftr)) / num
        return mean_l1
    
    def mean_L2(self, ftr):
        num = len(ftr)
        mean_l2 = np.sqrt(np.sum(ftr ** 2) / num)
        return mean_l2
      
    def get_global_ftr_dir(self):
        globalFtrDir = os.path.join(self.ori_img_folder, r'..', 'globalFtr')
        return globalFtrDir
    
    ''' compute image global feature
        reference: 2011 Learning Photographic Global Tonal Adjustment with a Database of Input Output Image Pairs '''
    def compute_image_global_ftr(self):
        print 'compute_image_global_ftr'
        if not self.do_compute_image_global_ftr:
            print 'compute_image_global_ftr:skip and return'
            return

        num_imgs = len(self.imgs)
        paras = {}
        paras['fredo_image_processing'] = self.fredo_image_processing
        paras['in_img_dir'] = self.ori_img_folder
        paras['log_luminance'] = self.log_luminance
        paras['semantic_map_dir'] = self.semantic_map_dir
        

        
        ''' parallel computing '''

        funcArgs = zip(self.imgs, [paras] * num_imgs)
        try:
            results = self.pool.map(get_image_global_ftr, funcArgs)
        except PatchPreparerError, e:
            print e
        results = zip(*results)
        lightness_hist, scene_brightness, cp1, cp2, cp3, cp4, \
        hl_clipping, L_spatial_distr, bgHueHist = results[0], results[1], \
        results[2], results[3], results[4], results[5], results[6], \
        results[7], results[8]

        lightness_hist = np.array(np.vstack(lightness_hist))
        scene_brightness = np.array(np.vstack(scene_brightness))
        cp1, cp2, cp3, cp4 = np.array(np.vstack(cp1)), np.array(np.vstack(cp2)), \
        np.array(np.vstack(cp3)), np.array(np.vstack(cp4))
        hl_clipping = np.array(np.vstack(hl_clipping))
        L_spatial_distr = np.array(np.vstack(L_spatial_distr))
        bgHueHist = np.array(np.vstack(bgHueHist))
        print 'lightness_hist shape,scene_brightness shape,cp1 shape, bgHueHist shape', \
        lightness_hist.shape, scene_brightness.shape, cp1.shape, bgHueHist.shape

        # apply PCA to control points        
        PCA_cp1 = PCA(cp1)
        cp1_mean, cp1_evals, cp1_evecs, cp1_cum_frac_val = PCA_cp1.get_mean_evals_evecs_cumFracEval()
        print 'cp1_cum_frac_val', cp1_cum_frac_val
        PCA_cp2 = PCA(cp2)
        cp2_mean, cp2_evals, cp2_evecs, cp2_cum_frac_val = PCA_cp2.get_mean_evals_evecs_cumFracEval()
        print 'cp2_cum_frac_val', cp2_cum_frac_val
        PCA_cp3 = PCA(cp3)
        cp3_mean, cp3_evals, cp3_evecs, cp3_cum_frac_val = PCA_cp3.get_mean_evals_evecs_cumFracEval()
        print 'cp3_cum_frac_val', cp3_cum_frac_val
        PCA_cp4 = PCA(cp4)
        cp4_mean, cp4_evals, cp4_evecs, cp4_cum_frac_val = PCA_cp4.get_mean_evals_evecs_cumFracEval()
        print 'cp4_cum_frac_val', cp4_cum_frac_val
        num_PCA_comp = 5
        cp1_pca = PCA_cp1.proj_topk(cp1, num_PCA_comp)
        cp2_pca = PCA_cp2.proj_topk(cp2, num_PCA_comp)
        cp3_pca = PCA_cp3.proj_topk(cp3, num_PCA_comp)
        cp4_pca = PCA_cp4.proj_topk(cp4, num_PCA_comp)
        
        ''' do L1/L2 normalization to different feature types '''
        mean_l2_lightness_hist = self.mean_L2(lightness_hist)
        lightness_hist_2 = lightness_hist / mean_l2_lightness_hist
        mean_l1_scene_brightness = self.mean_L1(scene_brightness)
        mean_l2_scene_brightness = self.mean_L2(scene_brightness)
        scene_brightness_1 = scene_brightness / mean_l1_scene_brightness
        scene_brightness_1 = scene_brightness_1.reshape((num_imgs, 1))
        scene_brightness_2 = scene_brightness / mean_l2_scene_brightness
        scene_brightness_2 = scene_brightness_2.reshape((num_imgs, 1))        
        mean_l1_cp1_pca = self.mean_L1(cp1_pca)
        mean_l2_cp1_pca = self.mean_L2(cp1_pca)        
        cp1_pca_1 = cp1_pca / mean_l1_cp1_pca
        cp1_pca_2 = cp1_pca / mean_l2_cp1_pca
        mean_l1_cp2_pca = self.mean_L1(cp2_pca)
        mean_l2_cp2_pca = self.mean_L2(cp2_pca)
        cp2_pca_1 = cp2_pca / mean_l1_cp2_pca
        cp2_pca_2 = cp2_pca / mean_l2_cp2_pca
        mean_l1_cp3_pca = self.mean_L1(cp3_pca)
        mean_l2_cp3_pca = self.mean_L2(cp3_pca)
        cp3_pca_1 = cp3_pca / mean_l1_cp3_pca
        cp3_pca_2 = cp3_pca / mean_l2_cp3_pca
        mean_l1_cp4_pca = self.mean_L1(cp4_pca)
        mean_l2_cp4_pca = self.mean_L2(cp4_pca)
        cp4_pca_1 = cp4_pca / mean_l1_cp4_pca
        cp4_pca_2 = cp4_pca / mean_l2_cp4_pca
        mean_l1_hl_clipping = self.mean_L1(hl_clipping)
        mean_l2_hl_clipping = self.mean_L2(hl_clipping)
        hl_clipping_1 = hl_clipping / mean_l1_hl_clipping
        hl_clipping_2 = hl_clipping / mean_l2_hl_clipping
        
        mean_l1_L_spatial_distr = self.mean_L1(L_spatial_distr)
        mean_l2_L_spatial_distr = self.mean_L2(L_spatial_distr)
        L_spatial_distr_1 = L_spatial_distr / mean_l1_L_spatial_distr
        L_spatial_distr_2 = L_spatial_distr / mean_l2_L_spatial_distr

        mean_l1_bgHueHist = self.mean_L1(bgHueHist)
        mean_l2_bgHueHist = self.mean_L2(bgHueHist)
        bgHueHist_1 = bgHueHist / mean_l1_bgHueHist if mean_l1_bgHueHist > 0 else bgHueHist
        bgHueHist_2 = bgHueHist / mean_l2_bgHueHist if mean_l2_bgHueHist > 0 else bgHueHist
            
        ''' img_global_ftr_l1 shape: n*d '''
        img_global_ftr_l1 = np.hstack((lightness_hist, scene_brightness_1, \
                              cp1_pca_1, cp2_pca_1, cp3_pca_1, cp4_pca_1, \
                              hl_clipping_1, L_spatial_distr_1, bgHueHist_1))
        print 'img_global_ftr_l1 shape', img_global_ftr_l1.shape
        mean_global_ftr_l1 = np.mean(img_global_ftr_l1, axis=0)
        img_global_ftr_l1 = img_global_ftr_l1 - mean_global_ftr_l1[np.newaxis, :]
        
        img_global_ftr_l2 = np.hstack((lightness_hist_2, scene_brightness_2, \
                              cp1_pca_2, cp2_pca_2, cp3_pca_2, cp4_pca_2, \
                              hl_clipping_2, L_spatial_distr_2, bgHueHist_2))
        print 'img_global_ftr_l2 shape', img_global_ftr_l2.shape
        mean_global_ftr_l2 = np.mean(img_global_ftr_l2, axis=0)
        img_global_ftr_l2 = img_global_ftr_l2 - mean_global_ftr_l2[np.newaxis, :]        
        
        globalFtrDir = self.get_global_ftr_dir()
        if not os.path.exists(globalFtrDir):
            os.mkdir(globalFtrDir)
        for i in range(len(self.imgs)):
            globalFtrPath = os.path.join(globalFtrDir, self.imgs[i])
            data = {}
            data['pix_global_ftr'] = img_global_ftr_l2[i, :]
            pickle(globalFtrPath, data)
    
    ''' if memory is large enough, load all input images into memory '''
    def do_preload_all_original_imgs(self, pool, paras):
        print 'do_preload_all_original_imgs'
        func_args = zip(self.tr_imgs, [paras] * len(self.tr_imgs))
        stTime = time.time()
        if not pool == None:
            ''' parallel computing '''
            try:
                res = pool.map(get_img_pixels, func_args)
            except PatchPreparerError, e:
                print e
        else:
            res = []
            for i in range(len(self.tr_imgs)):
                print '%d th image %s' % (i, self.tr_imgs[i])
                res += [get_img_pixels(func_args[i])]
        elapsedTm = time.time() - stTime
        print 'preload input images: %5.2f seconds' % elapsedTm
        in_imgs = res
        return in_imgs   
    
    ''' preload parsing/detection category-wise integral map for training images '''
    def do_preload_context_maps(self):
        print 'do_preload_context_maps'
        num_imgs = len(self.tr_imgs)
        context_maps = [None] * num_imgs
        for i in range(num_imgs):
            in_img_context_path = os.path.join\
            (self.sem_integral_map_dir, self.trSemIntegralMapFiles[i])
            context_map = scipy.io.loadmat(in_img_context_path)
            context_map = context_map['maps']
            context_map = context_map.reshape((self.local_context_paras['label_num']))
            context_maps[i] = context_map
        return context_maps
        
    ''' write batch files for training a NN to predict color mapping '''
    def write_color_batch_files(self):      
        print 'write_color_batch_files'
        if not self.do_write_color_batch_files:
            print 'write_color_batch_files: skip and return'
            return
        num_img = len(self.tr_imgs)
            
        paras = {}
        paras['in_img_dir'], paras['enh_img_dir'], paras['in_seg_dir'], \
        paras['semantic_map_dir'], paras['sem_integral_map_dir'], \
        paras['img_color_integral_dir'], paras['ori_img_edge_folder'], \
        paras['segment_random_sample_num'], paras['local_context_paras'], paras['fredo_image_processing'] = \
        self.ori_img_folder, self.enh_img_folder, self.img_seg_dir, \
        self.semantic_map_dir, self.sem_integral_map_dir, \
        self.img_color_integral_dir, self.ori_img_edge_folder, \
        self.segment_random_sample_num , self.local_context_paras, self.fredo_image_processing 
        if not self.img_color_integral_dir == None:
            paras['local_context_color_paras']=self.local_context_color_paras
        
        funcArgs = zip(self.tr_imgs, [paras] * num_img)
         
        stTime = time.time()
        
        ''' parallel processing '''
        try:
            results = self.pool.map(processImgs, funcArgs)  # 
        except PatchPreparerError, e:
            print e
        elapsedTm = time.time() - stTime
        print 'patch sampling are finished: %5.2f seconds' % elapsedTm
        ''' unzip a list of tuples '''
        results = zip(*results)
        ''' 'posxs' is a tuple '''
        posxs, posys, seg_in_pixs, seg_enh_pixs, pixContextSemFtrs = \
        results[0], results[1], results[2], results[3], results[4]
        if self.img_color_integral_dir:
            pix_context_color_ftrs, pix_color_hists = results[5], results[6] 
        
        posxs,posys,seg_in_pixs,seg_enh_pixs,pixContextSemFtrs=\
        list(posxs),list(posys),list(seg_in_pixs),list(seg_enh_pixs),list(pixContextSemFtrs)
        if self.img_color_integral_dir:
            pix_context_color_ftrs, pix_color_hists = \
            list(pix_context_color_ftrs),list(pix_color_hists)

        numsSample = [posx.shape[0] for posx in posxs] 
          
        for num in numsSample:
            assert num > 0
                              
        numAllSample = sum(numsSample)
        print 'totally %d samples' % numAllSample
        
        ''' randomly permutate patches within one image '''
        for k in range(num_img):
            pm = range(numsSample[k])
            rd.shuffle(pm)
            posxs[k], posys[k] = posxs[k][pm], posys[k][pm]
            assert numsSample[k] == seg_in_pixs[k].shape[0]
            seg_in_pixs[k] = seg_in_pixs[k][pm, :, :]
            seg_enh_pixs[k] = seg_enh_pixs[k][pm, :, :]
            pixContextSemFtrs[k] = pixContextSemFtrs[k][pm, :]
            if self.img_color_integral_dir:
                pix_context_color_ftrs[k] = pix_context_color_ftrs[k][pm, :]
                pix_color_hists[k] = pix_color_hists[k][pm, :]   
        
        pix_ftr_dim = CENTRAL_PX_FEATURE_DIM if CENTRAL_PX_FEATURE_DIM < 4 else next_4_multiple(CENTRAL_PX_FEATURE_DIM)
        pix_ftr_mean = np.zeros((pix_ftr_dim), dtype=np.single)        
        batchPixContextSemMean = np.zeros((self.local_context_paras['ftr_dim']), dtype=np.single)
        if self.img_color_integral_dir:        
            batchPixContextColorMean = np.zeros((self.local_context_paras['pool_region_num'] * 3), dtype=np.single)
            batchPixColorHistMean = np.zeros((2*self.local_context_color_paras['hist_bin_num']),\
                                             dtype=np.single)
        
        if self.preload_all_ori_img:
            print 'preload all input images'
            in_imgs = self.do_preload_all_original_imgs(self.pool, paras)
            print 'len in_imgs %d' % len(in_imgs)
        
        if not os.path.exists(self.data_save_path):
            os.makedirs(self.data_save_path)

        self.batch_sizes = [get_batch_size(num, self.num_batches) for num in numsSample]
        self.batch_start = [np.cumsum(np.hstack((np.array([0]), batchSizes[0:self.num_batches - 1])))\
                             for batchSizes in self.batch_sizes]
         
        print 'writing batch files'
        for j in range(self.num_batches):
            st_time_batch = time.time()
            print 'writing %d th batch' % (j + 1)
            batch_size = sum([batchSizes[j] for batchSizes in self.batch_sizes])
                
            batch_pix_ftr = np.zeros((pix_ftr_dim, batch_size), dtype=np.single)
            batch_pix_position_ftr = np.zeros((2, batch_size), dtype=np.single)

            batchInPixData = np.zeros((batch_size, self.segment_random_sample_num, 3), dtype=np.single)
            batchEnhPixData = np.zeros((self.segment_random_sample_num, 3, batch_size), dtype=np.single)
            '''shape:(n, self.segment_random_sample_num, 3)'''
            batchEnhPixDataView = batchEnhPixData.swapaxes(0, 2).swapaxes(1, 2)
            batchPixContextSem = np.zeros((self.local_context_paras['ftr_dim'], batch_size), dtype=np.single)
            batchPixContextSemView = batchPixContextSem.swapaxes(0, 1)
            
            if self.img_color_integral_dir:
                batchPixContextColor = np.zeros\
                ((self.local_context_paras['pool_region_num'] * 3, batch_size), dtype=np.single)
                batchPixColorHist = np.zeros\
                ((2*self.local_context_color_paras['hist_bin_num'], batch_size), dtype=np.single)
                batchPixContextColorView = batchPixContextColor.swapaxes(0, 1)
                batchPixColorHistView = batchPixColorHist.swapaxes(0, 1)
            
            batchPatchToImageName = [None] * batch_size
            pc = 0
            
            l_batch_sizes = np.array([self.batch_sizes[k][j] for k in range(num_img)])
            max_batch_size = np.max(l_batch_sizes)
            
            for k in range(num_img):
                if self.preload_all_ori_img:
                    in_img = in_imgs[k]
                else:
                    in_img = get_img_pixels([self.tr_imgs[k], paras])
                h, w = in_img.shape[0], in_img.shape[1]
                start, end = self.batch_start[k][j], self.batch_start[k][j] + self.batch_sizes[k][j]
                posx, posy = posxs[k][start:end], posys[k][start:end]
                normed_posx, normed_posy = np.single(posx) / np.single(w), np.single(posy) / np.single(h)

                ''' shape: n*ftr_dim '''
                l_pix_ftr = get_pixel_feature_v2(in_img, posx, posy)
                pix_ftr_dim_effective = l_pix_ftr.shape[1]
                batch_pix_ftr[:pix_ftr_dim_effective, pc:pc + len(posx)] = l_pix_ftr.transpose()
                batch_pix_position_ftr[:, pc:pc + len(posx)] = np.vstack((normed_posx, normed_posy))
                
                batchInPixData[pc:pc + len(posx), :, :] = seg_in_pixs[k][start:end, :, :]
                batchPixContextSemView[pc:pc + len(posx), :] = pixContextSemFtrs[k][start:end, :]
                if self.img_color_integral_dir:
                    batchPixContextColorView[pc:pc + len(posx), :] = pix_context_color_ftrs[k][start:end, :]
                    batchPixColorHistView[pc:pc + len(posx), :] = pix_color_hists[k][start:end, :]
                batchEnhPixDataView[pc:pc + len(posx), :, :] = seg_enh_pixs[k][start:end, :, :]
                for p in range(len(posx)):
                    batchPatchToImageName[pc+p]=self.tr_imgs[k]
                
                pc += len(posx)

            pix_ftr_mean += np.sum(batch_pix_ftr, axis=1)            
            batchPixContextSemMean += np.sum(batchPixContextSem, axis=1)
            if self.img_color_integral_dir:
                batchPixContextColorMean += np.sum(batchPixContextColor, axis=1)
                batchPixColorHistMean += np.sum(batchPixColorHist, axis=1)
            ''' randomly permute patches within a batch '''
            permu = range(batch_size)
            rd.shuffle(permu)
            
            batch_pix_ftr = batch_pix_ftr[:, permu]
            batch_pix_position_ftr = batch_pix_position_ftr[:, permu]
            batchInPixData = batchInPixData[permu, :, :]
            batchEnhPixData = batchEnhPixData[:, :, permu]
            batchPixContextSem = batchPixContextSem[:, permu]
            if self.img_color_integral_dir:
                batchPixContextColor = batchPixContextColor[:, permu]
                batchPixColorHist = batchPixColorHist[:, permu]
            batchPatchToImageName = [batchPatchToImageName[i] for i in permu]
            
            l_batch = {}
            l_batch['data'] = batch_pix_ftr
            l_batch['in_pix_pos'] = batch_pix_position_ftr
            l_batch['in_pix_data'] = batchInPixData
            batchEnhPixData = batchEnhPixData.reshape\
            ((self.segment_random_sample_num * 3, batch_size))
            l_batch['labels'] = batchEnhPixData
            l_batch['pixContextSemFtr'] = batchPixContextSem
            if self.img_color_integral_dir:
                l_batch['pixContextColor'] = batchPixContextColor
                l_batch['pixColorHist'] = batchPixColorHist
            l_batch['patch_to_image_name'] = batchPatchToImageName 
            st_time = time.time()
            pickle(os.path.join(self.data_save_path, 'data_batch_' + str(j + 1)), l_batch)
            pickle_batch_time = time.time() - st_time
            batch_time = time.time() - st_time_batch
            print 'pickle_batch_time %f batch_time %f' % (pickle_batch_time, batch_time)
        print 'finish'
        
        if self.preload_all_ori_img:
            del in_imgs
         
        st_time = time.time()
        
        pix_ftr_mean /= numAllSample            
        batchPixContextSemMean /= numAllSample
        if self.img_color_integral_dir:
            batchPixContextColorMean /= numAllSample
            batchPixColorHistMean /= numAllSample
            
        meta_dict = {}
        meta_dict['in_img_dir'] = self.ori_img_folder
        meta_dict['in_img_global_ftr_dir'] = self.get_global_ftr_dir()
        meta_dict['img_seg_dir'] = self.img_seg_dir
        meta_dict['sem_integral_map_dir'] = self.sem_integral_map_dir
        meta_dict['semantic_map_dir'] = self.semantic_map_dir
        if self.img_color_integral_dir:
            meta_dict['img_color_integral_dir'] = self.img_color_integral_dir
            meta_dict['pixContextColorMean'] = batchPixContextColorMean
            meta_dict['pixColorHistMean'] = batchPixColorHistMean
            meta_dict['local_context_color_paras'] = self.local_context_color_paras
        meta_dict['enh_img_dir'] = self.enh_img_folder
        
        meta_dict['tr_imgs'] = self.tr_imgs
        meta_dict['ts_imgs'] = self.ts_imgs
        meta_dict['imgs'] = self.imgs
        
        meta_dict['pos_x'] = posxs
        meta_dict['pos_y'] = posys
        meta_dict['parameters'] = paras
        meta_dict['local_context_paras'] = self.local_context_paras
        meta_dict['data_mean'] = pix_ftr_mean
        meta_dict['pixContextSemMean'] = batchPixContextSemMean
        meta_dict['num_vis'] = pix_ftr_dim
        meta_dict['img_size'] = 1
            
        meta_dict['num_colors'] = 3
    
        pickle(os.path.join(self.data_save_path, 'batches.meta'), meta_dict)
        elapsed_time = time.time() - st_time
        print 'elapsed time for writing meta:%f' % elapsed_time

        print 'exit write_color_batch_files'
        
    ''' write batch files for training a NN to predict edge gradient magnitude '''        
    def write_edge_pix_batch_files(self):
        print 'write_edge_pix_batch_files'
        if not self.do_write_edge_pix_batch_files:
            print 'write_edge_pix_batch_files:skip and return'
            return
        if self.num_proc > 1:
            pool = mtp.Pool(processes=self.num_proc)
        else:
            pool = None
                    
        num_imgs = len(self.tr_in_imgs)
        paras = {}
        paras['in_img_dir'], paras['enh_img_dir'], paras['in_img_context_dir'], \
        paras['ori_img_precomputed_context_ftr_dir'], \
        paras['ori_img_edge_folder'], paras['gaus_smooth_sigma'], \
        paras['local_context_paras'], paras['fredo_image_processing'] = \
        self.ori_img_folder, self.enh_img_folder, self.sem_integral_map_dir, \
        self.ori_img_precomputed_context_ftr_dir, \
        self.ori_img_edge_folder, self.gaus_smooth_sigma, \
        self.local_context_paras, self.fredo_image_processing
        
        st_time = time.time()
        func_args = zip(self.tr_in_imgs, self.tr_enh_imgs, self.trSemIntegralMapFiles, \
                      [paras] * num_imgs)
        if self.num_proc > 1:
            try:
                results = pool.map(process_img_edge_pix, func_args)
            except PatchPreparerError, e:
                print e
            results = zip(*results)
            edge_pixs_x, edge_pixs_y, grad_mags_in_img, grad_mags_enh_img, pix_local_contexts = \
            list(results[0]), list(results[1]), list(results[2]), list(results[3]), list(results[4])
        else:
            edge_pixs_x, edge_pixs_y, grad_mags_in_img, grad_mags_enh_img, pix_local_contexts = \
            [], [], [], [], []
            try:
                for i in range(num_imgs):
                    edge_pix_x, edge_pix_y, grad_mag_in_img, grad_mag_enh_img, pix_local_context = \
                    process_img_edge_pix(func_args[i])
                    edge_pixs_x += [edge_pix_x]
                    edge_pixs_y += [edge_pix_y]
                    grad_mags_in_img += [grad_mag_in_img]
                    grad_mags_enh_img += [grad_mag_enh_img]
                    pix_local_contexts += [pix_local_context]
            except PatchPreparerError, e:
                print e
        elapsed = time.time() - st_time
        print 'process_img_edge_pix elapsed time:%f' % elapsed
        
        num_pixs = [len(edge_pix_x) for edge_pix_x in edge_pixs_x]
        num_all_pixs = sum(num_pixs)
        print 'num_pixs', num_pixs
        print 'num_all_pixs', num_all_pixs       
        
        for i in range(num_imgs):
            rd_perm = range(num_pixs[i])
            rd.shuffle(rd_perm)
            edge_pixs_x[i] = edge_pixs_x[i][rd_perm]
            edge_pixs_y[i] = edge_pixs_y[i][rd_perm]
            grad_mags_in_img[i] = grad_mags_in_img[i][rd_perm]
            grad_mags_enh_img[i] = grad_mags_enh_img[i][rd_perm]
            pix_local_contexts[i] = pix_local_contexts[i][rd_perm, :]
        
        print 'num_edge_batches', self.num_edge_batches
        batch_sizes = [get_batch_size(num, self.num_edge_batches) for num in num_pixs]
        batch_start = [np.cumsum(np.hstack((np.array([0]), batch_size[:self.num_edge_batches - 1])))\
                       for batch_size in batch_sizes]

        pix_ftr_dim = CENTRAL_PX_FEATURE_DIM if CENTRAL_PX_FEATURE_DIM < 4 else next_4_multiple(CENTRAL_PX_FEATURE_DIM)
        pix_ftr_mean = np.zeros((pix_ftr_dim), dtype=np.single)
        batch_pix_local_context_mean = np.zeros((self.local_context_paras['ftr_dim']), dtype=np.single)

        if self.preload_all_ori_img:
            in_imgs = self.do_preload_all_original_imgs(pool, paras)
        
        if not os.path.exists(self.data_save_path_edge):
            os.makedirs(self.data_save_path_edge)
        
        preload_context_maps = False
        if preload_context_maps:
            tr_context_maps = self.do_preload_context_maps()
            
        
        st_time = time.time()
        print 'write edge pixel batches'
        for i in range(self.num_edge_batches):
            if i % 5 == 0:
                print '%d th out of %d batches' % (i, self.num_edge_batches)
            this_batch_size = sum([batch_size[i] for batch_size in batch_sizes])
            batch_pix_ftr = np.zeros((pix_ftr_dim, this_batch_size), dtype=np.single)
            batch_in_pix_gradmag = np.zeros((1, this_batch_size), dtype=np.single)
            batch_enh_pix_gradmag = np.zeros((1, this_batch_size), dtype=np.single)
            batch_pix_local_context = np.zeros((self.local_context_paras['ftr_dim'], this_batch_size), \
                                               dtype=np.single)
            batch_pix_to_imgID = np.zeros((this_batch_size), dtype=np.uint32)
            
            batch_pix_ftr_view = batch_pix_ftr.swapaxes(0, 1)
            batch_pix_local_context_view = batch_pix_local_context.swapaxes(0, 1)
            
            pc = 0
            
            for k in range(num_imgs):
                if self.preload_all_ori_img:
                    in_img = in_imgs[k]
                else:
                    in_img = get_img_pixels([self.tr_in_imgs[k], self.tr_enh_imgs[k], paras])
                
                h, w = in_img.shape[0], in_img.shape[1]
                start, end = batch_start[k][i], batch_start[k][i] + batch_sizes[k][i]
                posx, posy = edge_pixs_x[k][start:end], edge_pixs_y[k][start:end]
#                 print 'posx shape',posx.shape
                l_pix_ftr = get_pixel_feature_v2(in_img, posx, posy)
    ''' compute color integral map for computing window-wise mean color'''
    def compute_color_integral_map(self):
        if not self.do_compute_color_integral_map:
            print 'compute_color_integral_map : skip and return'
            return 
        if not self.img_color_integral_dir:
            print 'img_color_integral_dir is not specified. Return'
            return
        print 'compute_color_integral_map'
        if not os.path.exists(self.img_color_integral_dir):
            os.mkdir(self.img_color_integral_dir)
        os.chdir(self.img_color_integral_dir)
        
        st = time.time()
        num_img = len(self.imgs)
        im_paths = [os.path.join(self.ori_img_folder, f + '.tif') for f in self.imgs]
        sem_map_paths = [os.path.join(self.semantic_map_dir, f + '.mat') for f in self.imgs]
        
        funcArgs = zip(im_paths, sem_map_paths, [self.img_color_integral_dir] * num_img)
        self.pool.map(compute_color_integral_map_helper, funcArgs)
        elapsed = time.time() - st
        print 'computing color integral maps is completed. %4.3f seconds' % elapsed
        return
                        
    ''' compute mean color in each pooling region of contextual feature '''
    def precompute_local_context_color_ftr(self):
        if not self.do_precompute_local_context_mean_color_ftr:
            print 'precompute_local_context_color_ftr:skip and return'
            return
        if not self.precomputed_color_ftr_dir:
            print 'no precomputed_color_ftr_dir is specified,return'
            return
        print '-----precompute_local_context_color_ftr------'
        pool = mtp.Pool(processes=self.num_proc)
        
        if not os.path.exists(self.precomputed_color_ftr_dir):
            os.mkdir(self.precomputed_color_ftr_dir)
        os.chdir(self.precomputed_color_ftr_dir)
        num_img = len(self.in_imgs)
        
        apdWid = LOCAL_CONTEXT_MAP_APPEND_WIDTH
        
        for i in range(num_img):
            imId = i
            img_name = self.in_imgs[imId]
            mat_file_nm = os.path.join(self.precomputed_color_ftr_dir, \
                                       self.in_imgs[imId][:-4] + '_context_mean_color_ftr.mat')
            print 'process %d out of %d images %s' % \
            (i, num_img, img_name)
            if os.path.exists(mat_file_nm):
                print 'file %s already exists, skip' % mat_file_nm
                continue
            
            f = open(os.path.join(self.ori_img_folder, img_name), 'rb')
            tags = exifread.process_file(f, details=False)
            f.close()
            h_str = '%s' % tags['Image ImageLength']
            w_str = '%s' % tags['Image ImageWidth']
            h, w = int(h_str), int(w_str)
            
            colorIntegralFileNm = os.path.join(self.img_color_integral_dir, \
                                          self.colorIntegralFiles[imId])                        
            print 'load color integral map: %s' % colorIntegralFileNm
            color_itg_map = scipy.io.loadmat(colorIntegralFileNm)
            color_itg_map = color_itg_map['colorIntegralMaps']
            ''' color_itg_map shape: (3,h,w) '''        
            assert h == (color_itg_map.shape[1] - 2 * apdWid)
            assert w == (color_itg_map.shape[2] - 2 * apdWid)
            color_itg_map = color_itg_map.reshape\
            ((3, (h + 2 * apdWid), (w + 2 * apdWid)))  # L,a,b three channels
            
            context_color_ftrs = np.zeros((h * w, self.local_context_paras['pool_region_num'] * 3), \
                                    dtype=np.single)
            pix_x, pix_y = np.meshgrid(range(w), range(h))
            pix_x, pix_y = pix_x.flatten(), pix_y.flatten()
            st_time = time.time()
            get_pixel_local_context_mean_color\
            ([pix_x + apdWid, pix_y + apdWid, color_itg_map, \
              self.local_context_paras, pool, context_color_ftrs])            
            elapsed = time.time() - st_time
            print 'elapsed: %f' % (elapsed)
            
            context_color_ftrs = context_color_ftrs.reshape((h, w, self.local_context_paras['pool_region_num'] * 3))
            mat_dict = {}
            mat_dict['context_mean_color_ftr'] = context_color_ftrs
            scipy.io.savemat(mat_file_nm, mat_dict)
        pool.close()

    ''' compute pixel-wise local context features (25(or 28 if SPM is enabled) pooling regions, 20-bin histogram) '''
    def precompute_local_context_ftr(self):
        if not self.do_precompute_local_context_ftr:
            print 'precompute_local_context_ftr: skip and return'
            return
        if not self.ori_img_precomputed_context_ftr_dir:
            return
        print '-----precompute_local_context_ftr----'
        if not os.path.exists(self.ori_img_precomputed_context_ftr_dir):
            os.mkdir(self.ori_img_precomputed_context_ftr_dir)
        if 1:
            num_img = len(self.in_imgs)
        else:
            num_img = len(self.ts_img_id)
        pool = mtp.Pool(processes=self.num_proc)
        for i in range(num_img):
            if 1:
                imId = i
            else:                
               imId = self.ts_img_id[i]
                                
            img_name = self.in_imgs[imId]
            matFileNm = os.path.join(self.ori_img_precomputed_context_ftr_dir, self.in_imgs[imId][:-4] + '_context_ftr.mat')
                
            print 'process %d out of %d image %s' % \
            (i, num_img, img_name) 
           
            f = open(os.path.join(self.ori_img_folder, img_name), 'rb')
            tags = exifread.process_file(f, details=False)
            f.close()
            h_str = '%s' % tags['Image ImageLength']
            w_str = '%s' % tags['Image ImageWidth']
            h, w = int(h_str), int(w_str)
            coxtFileNm = os.path.join(self.sem_integral_map_dir, \
                                          self.semIntegralMapFiles[imId])
            print 'load context map: %s' % coxtFileNm
            context_map = scipy.io.loadmat(coxtFileNm)            
            context_map = context_map['maps'].reshape((self.local_context_paras['label_num']))

            assert h == (context_map[0].shape[0] - 2 * LOCAL_CONTEXT_MAP_APPEND_WIDTH)
            assert w == (context_map[0].shape[1] - 2 * LOCAL_CONTEXT_MAP_APPEND_WIDTH)
            print 'h,w', h, w
            conxtFtr = np.zeros((h * w, self.local_context_paras['ftr_dim']), dtype=np.single)
            pix_x, pix_y = np.meshgrid(range(w), range(h))
            pix_x, pix_y = pix_x.flatten(), pix_y.flatten()
            st = time.time()
            getPixLocContextV2\
            ([pix_x + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
              pix_y + LOCAL_CONTEXT_MAP_APPEND_WIDTH, context_map, \
              self.local_context_paras, pool, conxtFtr, self.num_proc])
            elapsed = time.time() - st
            print 'elapsed: %f' % (elapsed)
            conxtFtr = conxtFtr.reshape((h, w, self.local_context_paras['ftr_dim']))
            matDict = {}
            matDict['context_ftr'] = conxtFtr
            scipy.io.savemat(matFileNm, matDict)            
             
             
        pool.close()
        
    @classmethod
    def get_options_parser(cls):
        op = OptionsParser()
        op.add_option("num-proc", "num_proc", IntegerOptionParser, "number of parallel processes", \
                      default=1)
        op.add_option("ori-img-folder", "ori_img_folder", StringOptionParser, \
                      "original image folder", default="")
        op.add_option("img-seg-dir", "img_seg_dir", StringOptionParser, \
                      "folder of original image segmentations", default="")    
        op.add_option("semantic-map-dir", "semantic_map_dir", StringOptionParser, \
                      "folder of image parsing/detection results (non-integral map)", default="")         
        op.add_option("sem-integral-map-dir", "sem_integral_map_dir", StringOptionParser, \
                      "folder of original image parsing/detection results (integral map) ", default="")
        op.add_option("ori-img-precomputed-context-ftr-dir", "ori_img_precomputed_context_ftr_dir", \
                      StringOptionParser, "folder to save precomputed pixel local context feature", default="")
        op.add_option("img-color-integral-dir", "img_color_integral_dir", StringOptionParser, \
                      "folder to save original image context mean color results (integral map)", default="")        
        op.add_option("precomputed-color-ftr-dir", "precomputed_color_ftr_dir", \
                      StringOptionParser, "folder to save precomputed pixel local context color feature", default="")        
        op.add_option("ori-img-edge-folder", "ori_img_edge_folder", StringOptionParser, \
                      "folder of original image edge detection results", default="")        
        
        op.add_option("train-image-id-file", "train_image_id_file", StringOptionParser, \
                      "text file of training image id list", default="")
        op.add_option("test-image-id-file", "test_image_id_file", StringOptionParser, \
                      "text file of test image id list", default="")        
#         op.add_option("ori-img-saliency-folder", "ori_img_saliency_folder", StringOptionParser, \
#                       "original image saliency map folder", default="")  
        op.add_option("enh-img-folder", "enh_img_folder", StringOptionParser, \
                      "enhanced image folder", default="")
#         op.add_option("img-Cnn-ftr-file", "img_Cnn_ftr_file", StringOptionParser, "the path of image CNN feature file", \
#                       default='')
        op.add_option("data-save-path", "data_save_path", StringOptionParser, "path of saving folder", default="")
        op.add_option("data-save-path-edge", "data_save_path_edge", StringOptionParser, "path of saving folder for edge pixels", default="")
                             
#         op.add_option("pixel-feature-neighbor-half-size", "pixel_feature_neighbor_half_size", IntegerOptionParser,
#                       "half size of neighborhood to estimate pixel feature", default=3)
        # patch side length would be 2*x+1
#         op.add_option("cnn-patch-half-size", "cnn_patch_half_size", IntegerOptionParser, \
#                       "half size of patch in CNN ", default=16)
        # # neighborhood side length would be 2*x+1
#         op.add_option("color-mapping-neighbor-size", 'color_mapping_neighbor_size', IntegerOptionParser, \
#                       "size of neighborhood in color mapping estimation", default=2)
        op.add_option("color-mapping-residue-thres", "color_mapping_residue_thres", FloatOptionParser, "threshold of mean residue ||r||_2^2 in color mapping estimation", default=4.0)
#         op.add_option("stride", "stride", IntegerOptionParser, "stride", default=OptionExpression("cnn_patch_half_size"))
#         op.add_option("ls-lambda", "ls_lambda", FloatOptionParser, \
#                       "coefficient of the 2nd term in regularized least square regression", default=1e-1)
#         op.add_option("ls-atol", "ls_atol", FloatOptionParser, "tolerance coefficient a", default=1e-6)
#         op.add_option("outlier-label-thres", "outlier_label_thres", FloatOptionParser, \
#                        "threshold of multiple of standard deviation to detect outlier labels", default=6.0)
             
#         op.add_option("color-mapping-PCA-thres", "color_mapping_PCA_thres", FloatOptionParser, \
#                       "threshold of cumulative eigenvalues", default=.9)
        op.add_option("num-batches", 'num_batches', IntegerOptionParser, "number of batches", default=100)
        op.add_option("num-edge-batches", 'num_edge_batches', IntegerOptionParser,
                      "number of edge pixel batches", default=50)        
        op.add_option("preload-all-ori-img", 'preload_all_ori_img', BooleanOptionParser,
                      "enable it if memory is large enough to accommodate all original images", default=False)
        op.add_option("segment-random-sample-num", 'segment_random_sample_num', IntegerOptionParser,
                      "number of pixels to be randomly sampled within segment", default=25)
        op.add_option("gaus-smooth-sigma", "gaus_smooth_sigma", FloatOptionParser,
                      "sigma of Gaussian smoothing filter", default=1.5)
        op.add_option("fredo-image-processing", 'fredo_image_processing', BooleanOptionParser,
                      "increase exposure by 1.5 and normalize L channel", default=0)
        op.add_option("log-luminance", 'log_luminance', BooleanOptionParser,
                      "use log-luminance when computing image global feature?", default=0)  
        op.add_option("do-compute-image-global-ftr", 'do_compute_image_global_ftr', BooleanOptionParser,
                      "compute image global ftr?", default=0)         
        op.add_option("do-compute-color-integral-map", 'do_compute_color_integral_map', BooleanOptionParser,
                      "compute image color integral maps?", default=0) 
        op.add_option("do-precompute-local-context-mean-color-ftr",\
                      'do_precompute_local_context_mean_color_ftr', BooleanOptionParser,
                      "precompute image local context mean color feature", default=0)                 
        op.add_option("do-local-context-SPM-regions", 'do_local_context_SPM_regions', BooleanOptionParser,
                      "add 3 additional Sptial Pyramid Matching pooling regions to context feature", default=0)          
        
        op.add_option("do-precompute-local-context-ftr", 'do_precompute_local_context_ftr', BooleanOptionParser,
                      "precompute local context ftr?", default=0, requires=['ori_img_precomputed_context_ftr_dir'])         
        op.add_option("do-write-color-batch-files", 'do_write_color_batch_files', BooleanOptionParser,
                      "write color batch files?", default=0) 
        op.add_option("do-write-edge-pix-batch-files", 'do_write_edge_pix_batch_files', BooleanOptionParser,
                      "write edge pix batch files?", default=0)                                
        op.add_option("context-ftr-baseline", 'context_ftr_baseline', BooleanOptionParser,
                      "baseline implementation of context feature", default=0)                               
        return op
    
        
if __name__ == "__main__":
    op = TrainPatchPreparer.get_options_parser()
    op = TrainPatchPreparer.parse_options(op)
    preparer = TrainPatchPreparer(op)
    preparer.find_images()
    preparer.find_segments()
    preparer.find_semantic_maps()
    preparer.find_semantic_integral_files()
    preparer.compute_image_global_ftr()
    preparer.compute_color_integral_map()        
    preparer.find_color_integral_files()
    preparer.write_color_batch_files()
    # # preparer.write_edge_pix_batch_files()
    preparer.clear()
