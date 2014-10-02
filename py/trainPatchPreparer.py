import sys
sys.path.append('/home/yzc/yzc/proj/cuda-convnet-plus/cuda-convnet-plus/py')
sys.path.append('D:\yzc\proj\cuda-convnet-plus\cuda-convnet-plus\py')
sys.path.append('D:\proj\cuda-convnet-plus\cuda-convnet-plus\py')


from colorMapping import *
from options import *
from PrinCompAnal import *
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


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class PatchPreparerError(Exception):
    pass
    
class TrainPatchPreparer:
#     LOCAL_CONTEXT_MAP_APPEND_WIDTH = 100
    
    def __init__(self, op):
        self.op = op

        for o in op.get_options_list():
            setattr(self, o.name, o.value)        
        op.print_values()
        self.init_local_context_ftr_paras()
        
        if self.num_proc > 1:
            self.pool = mtp.Pool(processes=self.num_proc)
        else:
            self.pool = None
    
    def clear(self):
        if self.pool:
            self.pool.close()



    ''' initialize parameter setting for computing context feature'''
    ''' only 4 pooling regions are used '''
#     def init_local_context_ftr_paras(self):
#         context = {}
#         context['label_num'] = SEMANTIC_LABELS_NUM
#         context['radius'] = 3
#         context['layer_num'] = 4
#         
#         n_pooling_region = context['layer_num']
#         
#         offsets = np.zeros((4, n_pooling_region), dtype=np.int32)
#         area_pix_num = np.zeros((n_pooling_region), dtype=np.int32)
#         for i in range(context['layer_num']):
#             r = np.power(context['radius'], i + 1)
#             offsets[:, i] = [-r, -r, r, r]
#             area_pix_num[i] = (2 * r + 1) ** 2
# 
#         context['offsets'] = offsets
#         context['area_pix_num'] = area_pix_num
#         context['ftr_dim'] = offsets.shape[1] * context['label_num']
#         context['pool_region_num'] = offsets.shape[1]
# #         print 'context',context
#         self.local_context_paras = context


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
        
        
        
        self.local_context_paras = context
        
    '''    load precomputed global CNN image feature    '''
    def load_image_feature(self):
        if self.img_Cnn_ftr_file:
            print 'load image feature file: %s' % self.img_Cnn_ftr_file
            image_ftr = unpickle(self.img_Cnn_ftr_file)
            '''    shape: n * d    '''
            self.imgCnnFtr = image_ftr['data'] 
            image_ftr_mean = np.mean(self.imgCnnFtr, axis=0)
            self.imgCnnFtr = (self.imgCnnFtr - image_ftr_mean[np.newaxis, :]).transpose()
            assert self.imgCnnFtr.shape[1] >= len(self.in_imgs)            
        else:
            print 'global CNN image feature file is not provided'
            self.imgCnnFtr = np.zeros((0))
            image_ftr_mean = np.zeros((0))

    @staticmethod
    def parse_options(op):
        op.parse()
        op.eval_expr_defaults()
        return op
    
    
    def findSemanticMaps(self):
        if not self.semantic_map_dir:
            return
        print 'findSemanticMaps'
        self.semMapFiles = \
        [file for file in os.listdir(self.semantic_map_dir)\
         if os.path.isfile(os.path.join(self.semantic_map_dir, file))\
         and fnmatch.fnmatch(file, '*.mat')]


        #print self.semMapFiles
        if self.is_first_effect: 
            self.semMapFiles = sorted(self.semMapFiles, key = natural_key)
        else:
            self.semMapFiles = sorted(self.semMapFiles)

        assert len(self.semMapFiles) == len(self.in_imgs)
        
        self.trSemMapFiles = [self.semMapFiles[id] for id in self.tr_img_id]
        self.tsSemMapFiles = [self.semMapFiles[id] for id in self.ts_img_id]
        
        
    ''' find input and enhanced image pair.
        find train/test image ids'''
    def find_images(self):
        self.in_imgs = \
        [file for file in os.listdir(self.ori_img_folder)\
         if os.path.isfile(os.path.join(self.ori_img_folder, file))\
         and fnmatch.fnmatch(file, '*.tif')]
        self.enh_imgs = \
        [file for file in os.listdir(self.enh_img_folder)\
         if os.path.isfile(os.path.join(self.enh_img_folder, file))\
         and fnmatch.fnmatch(file, '*.tif')]
        
        self.in_imgs = sorted(self.in_imgs)
        self.enh_imgs = sorted(self.enh_imgs)
        print self.in_imgs[:5]

        #print self.in_imgs
        #print self.enh_imgs
        if not self.is_first_effect:
            self.tr_img_id = readImgNameFlie(self.train_image_id_file, self.in_imgs)
            self.ts_img_id = readImgNameFlie(self.test_image_id_file, self.in_imgs)
        else:
            self.tr_img_id = read_image_id_txt_file(self.train_image_id_file)
            self.ts_img_id = read_image_id_txt_file(self.test_image_id_file)
        assert (len(self.in_imgs) == len(self.enh_imgs))
        print "find %d original images" % len(self.in_imgs)        
        print '%d,%d training/test images' % (len(self.tr_img_id), len(self.ts_img_id))
        
        # compare file names
        for i in range(len(self.in_imgs)):
            img_name = self.in_imgs[i][:-4]
            if not self.in_imgs[i] == self.enh_imgs[i]:
                print "original:%s enhanced:%s:" % (self.in_imgs[i], self.enh_imgs[i])
                raise PatchPreparerError('inconsistent original/enhanced image files')         
            
        self.tr_in_imgs = [self.in_imgs[imId] for imId in self.tr_img_id ]
        self.tr_enh_imgs = [self.enh_imgs[imId] for imId in self.tr_img_id]
        self.ts_in_imgs = [self.in_imgs[imId] for imId in self.ts_img_id]
        self.ts_enh_imgs = [self.enh_imgs[imId] for imId in self.ts_img_id]

    
    ''' find precomputed segmentation files '''
    def find_segments(self):
        if self.img_seg_dir:
            self.in_seg_files = [fl for fl in os.listdir(self.img_seg_dir)\
                                  if os.path.isfile(os.path.join(self.img_seg_dir, fl)) and fnmatch.fnmatch(fl, "*.seg")]


        if not self.is_first_effect:
            self.in_seg_files = sorted(self.in_seg_files)            
            assert len(self.in_imgs) == len(self.in_seg_files)
            for i in range(len(self.in_imgs)):
                if not self.in_imgs[i][0:-4] == self.in_seg_files[i][0:-4]:
                    print "original:%s original seg:%s:" % (self.in_imgs[i], self.in_seg_files[i])
                    raise PatchPreparerError('inconsistent original segmentation files')
            self.tr_in_seg_file = [self.in_seg_files[id] for id in self.tr_img_id ]
        else:
            self.in_seg_files = [None] * len(self.in_imgs)
            self.tr_in_seg_file = [None] * len(self.tr_in_imgs)

            
    ''' find precomputed semantic integral maps'''
    def findSemIntegralFiles(self):
        if self.sem_integral_map_dir:
            self.semIntegralMapFiles = [file for file in os.listdir(self.sem_integral_map_dir)\
                                   if os.path.isfile(os.path.join(self.sem_integral_map_dir, file))\
                                   and fnmatch.fnmatch(file, "*.mat")]
            if self.is_first_effect:
                self.semIntegralMapFiles = sorted(self.semIntegralMapFiles, key = natural_key)
            else:
                self.semIntegralMapFiles = sorted(self.semIntegralMapFiles)
            print self.semIntegralMapFiles[:5]
            assert len(self.in_imgs) == len(self.semIntegralMapFiles)            
            self.trSemIntegralMapFiles = [self.semIntegralMapFiles[id] for id in self.tr_img_id]
            self.tsSemIntegralMapFiles = [self.semIntegralMapFiles[id] for id in self.ts_img_id]

    ''' find precomputed color integral maps '''
    def findColorIntegralFiles(self):
        if self.img_color_integral_dir:
            self.colorIntegralFiles = [file for file in os.listdir(self.img_color_integral_dir)\
                                   if os.path.isfile(os.path.join(self.img_color_integral_dir, file))\
                                   and fnmatch.fnmatch(file, "*.mat")]
            if not self.is_first_effect:
                self.colorIntegralFiles = sorted(self.colorIntegralFiles)
            assert len(self.in_imgs) == len(self.colorIntegralFiles)
            self.trColorIntegralFiles = [self.colorIntegralFiles[id] for id in self.tr_img_id]
            self.tsColorIntegralFiles = [self.colorIntegralFiles[id] for id in self.ts_img_id]             
        else:
            self.trColorIntegralFiles = [None] * len(self.tr_img_id)
            self.tsColorIntegralFiles = [None] * len(self.ts_img_id)

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
      
    def getGlobalFtrDir(self):
        globalFtrDir=os.path.join(self.ori_img_folder,r'..','globalFtr')
        return globalFtrDir
    ''' compute image global feature
        reference: 2011 Learning Photographic Global Tonal Adjustment with a Database of Input Output Image Pairs '''
    def compute_image_global_ftr(self):
        print 'compute_image_global_ftr'
        if not self.do_compute_image_global_ftr:
            print 'compute_image_global_ftr:skip and return'
            return
#         num_imgs = 100
        num_imgs = len(self.in_imgs)
        paras = {}
        paras['fredo_image_processing'] = self.fredo_image_processing
        paras['in_img_dir'] = self.ori_img_folder
        paras['log_luminance'] = self.log_luminance
        paras['semantic_map_dir'] = self.semantic_map_dir
        
        if self.sem_fgbg_matfile:
            mat = scipy.io.loadmat(self.sem_fgbg_matfile)
            mat = mat['xpro_table'].transpose()
            print mat.shape
            
            assert mat.shape[0] == num_imgs
            semFgBg = np.zeros((num_imgs, SEMANTIC_LABELS_NUM))
            semFgBg[:, :(SEMANTIC_LABELS_NUM - 1)] = mat
        
        ''' parallel computing '''
        if self.num_proc > 1:
            pool = mtp.Pool(processes=self.num_proc)
            if self.sem_fgbg_matfile:
                funcArgs = zip(self.in_imgs, [paras] * num_imgs, self.semMapFiles, semFgBg)
            else:
                funcArgs = zip(self.in_imgs, [paras] * num_imgs)
            try:
                results = pool.map(get_image_global_ftr, funcArgs)
            except PatchPreparerError, e:
                print e
            results = zip(*results)
            lightness_hist, scene_brightness, cp1, cp2, cp3, cp4, \
            hl_clipping, L_spatial_distr, bgHueHist = results[0], results[1], \
            results[2], results[3], results[4], results[5], results[6], \
            results[7], results[8]
            pool.close()     
        else:
            lightness_hist, scene_brightness, cp1, cp2, cp3, cp4, \
            hl_clipping, L_spatial_distr, bgHueHist = [], [], [], [], [], [], [], [], []
            for i in range(num_imgs):
                if i % 20 == 0:
                    print '%d out of %d images' % (i, num_imgs)
                if self.sem_fgbg_matfile:
                    l_funcArg = [self.in_imgs[i], paras, self.semMapFiles[i], semFgBg[i]]
                else:
                    l_funcArg = [self.in_imgs[i], paras]
                l_lightness_hist, l_scene_brightness, l_cp1, l_cp2, l_cp3, l_cp4, \
                l_hl_clipping, l_L_spatial_distr, l_bgHueHist = get_image_global_ftr(l_funcArg)
                lightness_hist += [l_lightness_hist]
                scene_brightness += [l_scene_brightness]
                cp1 += [l_cp1]
                cp2 += [l_cp2]
                cp3 += [l_cp3]
                cp4 += [l_cp4]
                hl_clipping += [l_hl_clipping]
                L_spatial_distr += [l_L_spatial_distr]
                bgHueHist += [l_bgHueHist]
        lightness_hist = np.array(np.vstack(lightness_hist))
        scene_brightness = np.array(np.vstack(scene_brightness))
        cp1, cp2, cp3, cp4 = np.array(np.vstack(cp1)), np.array(np.vstack(cp2)), \
        np.array(np.vstack(cp3)), np.array(np.vstack(cp4))
        hl_clipping = np.array(np.vstack(hl_clipping))
        L_spatial_distr = np.array(np.vstack(L_spatial_distr))
        bgHueHist = np.array(np.vstack(bgHueHist))
        print 'lightness_hist shape,scene_brightness shape,cp1 shape, bgHueHist shape', \
        lightness_hist.shape, scene_brightness.shape, cp1.shape, bgHueHist.shape

        # do PCA to control points        
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
        
        if 0:
            print 'lightness_hist shape', lightness_hist.shape
            print 'scene_brightness shape', scene_brightness.shape
            print 'cp1_pca shape', cp1_pca.shape
            print 'hl_clipping shape', hl_clipping.shape
            print 'L_spatial_distr shape', L_spatial_distr.shape
            print 'bgHueHist shape',bgHueHist.shape
            
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
        
        globalFtrDir=self.getGlobalFtrDir()
        if not os.path.exists(globalFtrDir):
            os.mkdir(globalFtrDir)
        for i in range(len(self.in_imgs)):
            globalFtrPath=os.path.join(globalFtrDir,self.in_imgs[i][:-4])
            data={}
            data['pix_global_ftr']=img_global_ftr_l2[i,:]
            pickle(globalFtrPath,data)
        
#         fn = os.path.join(self.ori_img_folder, 'pix_global_ftr')
#         data = {}
#         data['img_global_ftr_l1'] = img_global_ftr_l1.transpose()
#         data['img_global_ftr_l2'] = img_global_ftr_l2.transpose()
#         pickle(fn, data)
    
    ''' check if some image has no patches sampled '''    
    def zero_patch_image(self, num_patches):
        zeroPatchImg = [ 0 if num > 0 else 1 for num in num_patches]
        zeroPatchImgIdx = np.nonzero(zeroPatchImg)[0]
        if len(zeroPatchImgIdx) > 0:
            print '%d zero-patch images' % len(zeroPatchImgIdx)
            print zeroPatchImgIdx
            for i in zeroPatchImgIdx:
                print 'img:%s' % self.tr_in_imgs[i]
        numZeroPatchImg = sum(zeroPatchImg)
        if numZeroPatchImg > 0:
            print '%d images do not have patch sampled' % numZeroPatchImg
            sys.exit(0)  
    
#     def visualize_samping_positions(self, posXs, posYs):
#         # visualize sampling positions
#         numImg = len(self.tr_in_imgs)
#         for k in range(numImg):
#             in_img_path = self.tr_in_imgs[k]
#             in_img = (scipy.ndimage(in_img_path))
#             for p in range(len(posXs[k])):
#                 px, py = posXs[k][p], posYs[k][p]
#                 in_img[py - 3:py + 4, px - 3:px + 4, 1:3] = 0
#             mpplot.figure(k)
#             mpplot.imshow(in_img)
#             mpplot.show()    
    
    ''' if memory is large enough, load all input images into memory '''
    def do_preload_all_original_imgs(self, pool, paras):
        print 'do_preload_all_original_imgs'
        func_args = zip(self.tr_in_imgs, self.tr_enh_imgs, \
                        [paras] * len(self.tr_in_imgs))
        stTime = time.time()
        if not pool == None:
            ''' parallel computing '''
            try:
                res = pool.map(get_img_pixels, func_args)
            except PatchPreparerError, e:
                print e
        else:
            res = []
            for i in range(len(self.tr_in_imgs)):
                print '%d th image %s' % (i + 1, self.tr_in_imgs[i])
                res += [get_img_pixels(func_args[i])]
        elapsedTm = time.time() - stTime
        print 'preload input images: %5.2f seconds' % elapsedTm
        in_imgs = res
        return in_imgs   
    
    ''' preload parsing/detection category-wise integral map for training images '''
    def do_preload_context_maps(self):
        print 'do_preload_context_maps'
        num_imgs = len(self.tr_in_imgs)
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
        ''' flag if we compute 1) mean colors for each of 25 contextual pooling regions
            2) color histogram in a,b channels for a local window centered at the segment
        '''
#         local_context_color = False
        
        print 'write_color_batch_files'
        if not self.do_write_color_batch_files:
            print 'write_color_batch_files: skip and return'
            return
        numImg = len(self.tr_in_imgs)
        
        
        if self.num_proc > 1:
            pool = mtp.Pool(processes=self.num_proc)
        else:
            pool = None
            
        paras = {}
        paras['in_img_dir'], paras['enh_img_dir'], paras['in_seg_dir'], \
        paras['semantic_map_dir'], paras['sem_integral_map_dir'], \
        paras['img_color_integral_dir'], paras['ori_img_edge_folder'], paras['patch_half_size'], \
        paras['nbSize'], paras['stride'], paras['nb_hs'], \
        paras['segment_random_sample_num'], paras['local_context_paras'], paras['fredo_image_processing'] = \
        self.ori_img_folder, self.enh_img_folder, self.img_seg_dir, \
        self.semantic_map_dir, self.sem_integral_map_dir, \
        self.img_color_integral_dir, self.ori_img_edge_folder, self.cnn_patch_half_size, \
        self.color_mapping_neighbor_size, self.stride, self.pixel_feature_neighbor_half_size, \
        self.segment_random_sample_num , self.local_context_paras, self.fredo_image_processing        
        
#         if local_context_color:
#             local_context_color_paras = {}
#             local_context_color_paras['hf_sz'], local_context_color_paras['hist_bin_num'] = 50, 32
#             paras['local_context_color_paras'] = local_context_color_paras
            
        funcArgs = zip(self.tr_in_imgs, self.tr_enh_imgs, self.tr_in_seg_file, \
                        self.trSemMapFiles, self.trSemIntegralMapFiles, self.trColorIntegralFiles, [paras] * numImg)
        
        stTime = time.time()
        if self.num_proc > 1:
            ''' parallel processing '''
            try:
                results = pool.map(processImgs, funcArgs)  # 
            except PatchPreparerError, e:
                print e
            elapsedTm = time.time() - stTime
            print 'patch sampling and color mapping estimation are finished: %5.2f seconds' % elapsedTm
            ''' unzip a list of tuples '''
            results = zip(*results)
            ''' 'posxs' is a tuple '''
            ori_pos_nums, posxs, posys, seg_in_pixs, seg_enh_pixs, pixContextSemFtrs = \
            results[0], results[1], results[2], results[3], results[4], results[5],  
            if self.img_color_integral_dir:
                pixContextColorFtrs, pixContextPixnumFtrs = results[6], results[7] 
        else:
            ''' sequential processing '''
            ori_pos_nums, posxs, posys, seg_in_pixs, seg_enh_pixs, pixContextSemFtrs = \
            [], [], [], [], [], []
            if self.img_color_integral_dir:
                pixContextColorFtrs, pixContextPixnumFtrs = [], []
            try:
                for i in range(numImg):
                    print '%d th out of %d images:%s' % (i, numImg, self.tr_in_imgs[i])
                    if self.img_color_integral_dir:
                        ori_pos_num, posx, posy, seg_in_pix, seg_enh_pix, pixContextSemFtr= \
                        processImgs(funcArgs[i])
                    else:
                        ori_pos_num, posx, posy, seg_in_pix, seg_enh_pix, pixContextSemFtr= \
                        processImgs(funcArgs[i])
                    
                    ori_pos_nums += [ori_pos_num]
                    posxs += [posx]
                    posys += [posy]
                    seg_in_pixs += [seg_in_pix]
                    seg_enh_pixs += [seg_enh_pix]
                    pixContextSemFtrs += [pixContextSemFtr]
                    if self.img_color_integral_dir:
                        pixContextColorFtrs += [pixContextColorFtr]
                        pixContextPixnumFtrs += [pixContextPixnumFtr]
                    
            except PatchPreparerError, e:
                print e

        posxs, posys, seg_in_pixs, seg_enh_pixs, pixContextSemFtrs = \
        np.array(posxs), np.array(posys), np.array(seg_in_pixs), np.array(seg_enh_pixs), \
        np.array(pixContextSemFtrs) 
        if self.img_color_integral_dir:
            pixContextColorFtrs, pixContextPixnumFtrs = np.array(pixContextColorFtrs), np.array(pixContextPixnumFtrs)
        numPch = [posx.shape[0] for posx in posxs] 

        if 0:
            ''' visualize sampled segments position '''
            vis_sampling_dir = '/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/vis_sampling'
            if not os.path.exists(vis_sampling_dir):
                os.mkdir(vis_sampling_dir)
            for i in range(len(self.tr_sc_maps)):
                in_img_sc_map_path = os.path.join(self.ori_img_saliency_folder, self.tr_sc_maps[i])
                in_img_saliency = scipy.misc.imread(in_img_sc_map_path)
                l_h, l_w = in_img_saliency.shape[0], in_img_saliency.shape[1]
                in_img_saliency_rgb = np.zeros((l_h, l_w, 3), dtype=np.single)
                in_img_saliency_rgb[:, :, :] = (np.single(in_img_saliency) / 255.0)[:, :, np.newaxis]
                in_img_saliency_rgb[posys[i], posxs[i], :] = [1.0, 0, 0]
                scipy.misc.imsave(os.path.join(vis_sampling_dir, self.tr_in_imgs[i],), in_img_saliency_rgb)            

        self.zero_patch_image(numPch)
                  
        numAllPch = sum(numPch)
        print 'totally %d patches' % numAllPch
        
        ''' randomly permutate patches within one image '''
        for k in range(numImg):
            pm = range(numPch[k])
            rd.shuffle(pm)
            posxs[k], posys[k] = posxs[k][pm], posys[k][pm]
#             self.labels[k] = self.labels[k][:, pm]
            assert numPch[k] == seg_in_pixs[k].shape[0]
            seg_in_pixs[k] = seg_in_pixs[k][pm, :, :]
            seg_enh_pixs[k] = seg_enh_pixs[k][pm, :, :]
            pixContextSemFtrs[k] = pixContextSemFtrs[k][pm, :]
            if self.img_color_integral_dir:
                pixContextColorFtrs[k] = pixContextColorFtrs[k][pm, :]
                pixContextPixnumFtrs[k] = pixContextPixnumFtrs[k][pm, :]
                  
        ''' plot histogram of number of sampled patches '''
        # mpplot.figure()
        # mpplot.hist(numPch, 100)
        # mpplot.savefig('number of sample patches.png')        
        

        patchH, patchW = 2 * self.cnn_patch_half_size + 1, \
        2 * self.cnn_patch_half_size + 1    
        
        pix_ftr_dim = CENTRAL_PX_FEATURE_DIM if CENTRAL_PX_FEATURE_DIM < 4 else next_4_multiple(CENTRAL_PX_FEATURE_DIM)
        patch_dim = 3
        print 'pix_ftr_dim:%d' % pix_ftr_dim
        if self.patch_data == 1:
            patch_mean = np.zeros((patch_dim, patchH, patchW), dtype=np.float32)
            patch_mean_view = patch_mean.swapaxes(0, 1).swapaxes(1, 2)  # dim: patchH * patchW * patchCh
        
        pix_ftr_mean = np.zeros((pix_ftr_dim), dtype=np.single)        
        batchPixContextSemMean = np.zeros((self.local_context_paras['ftr_dim']), dtype=np.single)
        if self.img_color_integral_dir:        
            batchPixContextColorMean = np.zeros((self.local_context_paras['pool_region_num'] * 3), dtype=np.single)
            batchPixContextPixNumMean = np.zeros((self.local_context_paras['pool_region_num']), dtype=np.single)
        
        if self.preload_all_ori_img:
            in_imgs = self.do_preload_all_original_imgs(pool, paras)
            print 'len in_imgs  in_imgs 0 shape', len(in_imgs), in_imgs[0].shape
        
        if not os.path.exists(self.data_save_path):
            os.makedirs(self.data_save_path)

        self.batch_sizes = [get_batch_size(num, self.num_batches) for num in numPch]
        self.batch_start = [np.cumsum(np.hstack((np.array([0]), batchSizes[0:self.num_batches - 1])))\
                             for batchSizes in self.batch_sizes]
         
        print 'writing batch files'
        for j in range(self.num_batches):
            st_time_batch = time.time()
            print 'writing %d th batch' % (j + 1)
            batch_size = sum([batchSizes[j] for batchSizes in self.batch_sizes])
            if self.patch_data == 1:
                batch_patch_data = np.zeros((patch_dim, patchH, patchW, batch_size), dtype=np.single)
                batch_patch_data_view = np.swapaxes(batch_patch_data, 0, 3)  # shape: n*h*w*ch
            
            batch_pix_ftr = np.zeros((pix_ftr_dim, batch_size), dtype=np.single)
            batch_pix_position_ftr = np.zeros((2, batch_size), dtype=np.single)

            batchInPixData = np.zeros((batch_size, self.segment_random_sample_num, 3), dtype=np.single)
            batchEnhPixData = np.zeros((self.segment_random_sample_num, 3, batch_size), dtype=np.single)
            batchPixContextSem = np.zeros((self.local_context_paras['ftr_dim'], batch_size), dtype=np.single)
            batchPixContextSemView = batchPixContextSem.swapaxes(0, 1)
            
            if self.img_color_integral_dir:
                batchPixContextColor = np.zeros((self.local_context_paras['pool_region_num'] * 3, batch_size), dtype=np.single)
                batchPixContextPixNum = np.zeros((self.local_context_paras['pool_region_num'], batch_size), dtype=np.single)
                batchPixContextColorView = batchPixContextColor.swapaxes(0, 1)
                batchPixContextPixNumView = batchPixContextPixNum.swapaxes(0, 1)
            batchEnhPixDataView = batchEnhPixData.swapaxes(0, 2).swapaxes(1, 2)  # shape:(n, self.segment_random_sample_num, 3)

            
            batchPatchToImageID = np.zeros((batch_size), dtype=np.uint32)
            pc = 0
            
            l_batch_sizes = np.array([self.batch_sizes[k][j] for k in range(numImg)])
            max_batch_size = np.max(l_batch_sizes)
            ''' preallocate memory for reuse '''
            if self.patch_data == 1:
                in_patches = np.zeros((max_batch_size, patchH, patchW, patch_dim), dtype=np.single)
            
            for k in range(numImg):
                if self.preload_all_ori_img:
                    in_img = in_imgs[k]
                else:
                    in_img = get_img_pixels([self.tr_in_imgs[k], self.tr_enh_imgs[k], paras])
                h, w = in_img.shape[0], in_img.shape[1]
                start, end = self.batch_start[k][j], self.batch_start[k][j] + self.batch_sizes[k][j]
                posx, posy = posxs[k][start:end], posys[k][start:end]
                normed_posx, normed_posy = np.single(posx) / np.single(w), np.single(posy) / np.single(h)
                if self.patch_data == 1:
                    get_patches(in_img, posx, posy, self.cnn_patch_half_size, in_patches[0:end - start, :, :, :])
                    batch_patch_data_view[pc:pc + len(posx), :, :, :] = in_patches[0:end - start, :, :, :]

                # shape: n*ftr_dim
                l_pix_ftr = get_pixel_feature_v2(in_img, posx, posy, self.pixel_feature_neighbor_half_size)
                pix_ftr_dim_effective = l_pix_ftr.shape[1]
                batch_pix_ftr[:pix_ftr_dim_effective, pc:pc + len(posx)] = l_pix_ftr.transpose()
                batch_pix_position_ftr[:, pc:pc + len(posx)] = np.vstack((normed_posx, normed_posy))
                
                batchInPixData[pc:pc + len(posx), :, :] = seg_in_pixs[k][start:end, :, :]
                batchPixContextSemView[pc:pc + len(posx), :] = pixContextSemFtrs[k][start:end, :]
                if self.img_color_integral_dir:
                    batchPixContextColorView[pc:pc + len(posx), :] = pixContextColorFtrs[k][start:end, :]
                    batchPixContextPixNumView[pc:pc + len(posx), :] = pixContextPixnumFtrs[k][start:end, :]
                batchEnhPixDataView[pc:pc + len(posx), :, :] = seg_enh_pixs[k][start:end, :, :]
                batchPatchToImageID[pc:pc + len(posx)] = self.tr_img_id[k]
                
                pc += len(posx)
            if self.patch_data == 1:
                patch_mean_view += np.sum(batch_patch_data_view, axis=0)

            pix_ftr_mean += np.sum(batch_pix_ftr, axis=1)            
            batchPixContextSemMean += np.sum(batchPixContextSem, axis=1)
            if self.img_color_integral_dir:
                batchPixContextColorMean += np.sum(batchPixContextColor, axis=1)
                batchPixContextPixNumMean += np.sum(batchPixContextPixNum, axis=1)
            # randomly permute patches within a batch
            permu = range(batch_size)
            rd.shuffle(permu)
            if self.patch_data == 1:
                batch_patch_data = batch_patch_data[:, :, :, permu]
            
            batch_pix_ftr = batch_pix_ftr[:, permu]
            batch_pix_position_ftr = batch_pix_position_ftr[:, permu]
            batchInPixData = batchInPixData[permu, :, :]
            batchEnhPixData = batchEnhPixData[:, :, permu]
            batchPixContextSem = batchPixContextSem[:, permu]
            if self.img_color_integral_dir:
                batchPixContextColor = batchPixContextColor[:, permu]
                batchPixContextPixNum = batchPixContextPixNum[:, permu]
            batchPatchToImageID = batchPatchToImageID[permu]
            
            if self.patch_data == 1:
                batch_patch_data = batch_patch_data.reshape((patch_dim * patchH * patchW, batch_size))
            
            l_batch = {}
            if self.patch_data == 1:
                l_batch['patch_data'] = batch_patch_data
            l_batch['data'] = batch_pix_ftr
            l_batch['in_pix_pos'] = batch_pix_position_ftr
            l_batch['in_pix_data'] = batchInPixData
            batchEnhPixData = batchEnhPixData.reshape((self.segment_random_sample_num * 3, batch_size))
            l_batch['labels'] = batchEnhPixData
            l_batch['pixContextSemFtr'] = batchPixContextSem
            if self.img_color_integral_dir:
                l_batch['pixContextColor'] = batchPixContextColor
                l_batch['pixContextPixNum'] = batchPixContextPixNum
            l_batch['patch_to_imageID'] = batchPatchToImageID 
            st_time = time.time()
            pickle(os.path.join(self.data_save_path, 'data_batch_' + str(j + 1)), l_batch)
            pickle_batch_time = time.time() - st_time
            batch_time = time.time() - st_time_batch
            print 'pickle_batch_time %f batch_time %f' % (pickle_batch_time, batch_time)
        print 'finish'
        
        if self.preload_all_ori_img:
            del in_imgs
         
        st_time = time.time()
        print 'writing batch meta'
        if self.patch_data == 1:
            patch_mean_view /= numAllPch
        
        pix_ftr_mean /= numAllPch            
        batchPixContextSemMean /= numAllPch
        if self.img_color_integral_dir:
            batchPixContextColorMean /= numAllPch
            batchPixContextPixNumMean /= numAllPch
        
        if self.patch_data == 1:
            patch_mean = patch_mean.reshape((patch_dim * patchH * patchW, 1))
        meta_dict = {}
        meta_dict['in_img_dir'] = self.ori_img_folder
        meta_dict['in_img_global_ftr_dir']=self.getGlobalFtrDir()
        meta_dict['img_seg_dir'] = self.img_seg_dir
        meta_dict['sem_integral_map_dir'] = self.sem_integral_map_dir
        meta_dict['semantic_map_dir'] = self.semantic_map_dir
        if self.img_color_integral_dir:
            meta_dict['img_color_integral_dir'] = self.img_color_integral_dir
            meta_dict['pixContextColorMean'] = batchPixContextColorMean
            meta_dict['pixContextPixNumMean'] = batchPixContextPixNumMean
            meta_dict['colorIntegralFiles'] = self.colorIntegralFiles
        meta_dict['enh_img_dir'] = self.enh_img_folder
        
        meta_dict['semMapFiles'] = self.semMapFiles
        meta_dict['semIntegralMapFiles'] = self.semIntegralMapFiles
        
        meta_dict['tr_img_id'] = self.tr_img_id
        meta_dict['ts_img_id'] = self.ts_img_id
        meta_dict['in_imgs'] = self.in_imgs
        meta_dict['enh_imgs'] = self.enh_imgs
        
        meta_dict['pos_x'] = posxs
        meta_dict['pos_y'] = posys
        meta_dict['parameters'] = paras
        meta_dict['local_context_paras'] = self.local_context_paras
        meta_dict['pixel_feature_neighbor_half_size'] = self.pixel_feature_neighbor_half_size
        
        if self.patch_data == 1:
            meta_dict['patch_data_mean'] = np.single(patch_mean)

        meta_dict['data_mean'] = pix_ftr_mean
        meta_dict['pixContextSemMean'] = batchPixContextSemMean
        

        meta_dict['imgCnnFtr'] = self.imgCnnFtr
        if self.patch_data == 1:
            meta_dict['patch_num_vis'] = patch_dim * patchH * patchW
            meta_dict['patch_img_size'] = patchH

        meta_dict['num_vis'] = pix_ftr_dim
        meta_dict['img_size'] = 1
            
        meta_dict['num_colors'] = 3
    
        pickle(os.path.join(self.data_save_path, 'batches.meta'), meta_dict)
        elapsed_time = time.time() - st_time
        print 'elapsed time for writing meta:%f' % elapsed_time

        if self.num_proc > 1:
            pool.close()
        print 'exit write_color_batch_files'
#         mpplot.show()                        
        
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
        paras['nb_hs'], \
        paras['local_context_paras'], paras['fredo_image_processing'] = \
        self.ori_img_folder, self.enh_img_folder, self.sem_integral_map_dir, \
        self.ori_img_precomputed_context_ftr_dir, \
        self.ori_img_edge_folder, self.gaus_smooth_sigma, \
        self.pixel_feature_neighbor_half_size, \
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
#         edge_pixs_x, edge_pixs_y, grad_mags_in_img, grad_mags_enh_img = \
#         np.int32(np.array(edge_pixs_x)), np.int32(np.array(edge_pixs_y)), \
#         np.array(grad_mags_in_img), np.array(grad_mags_enh_img)
        
        for i in range(num_imgs):
            rd_perm = range(num_pixs[i])
            rd.shuffle(rd_perm)
#             if i == 0:
#                 print 'rd_perm', rd_perm
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
                l_pix_ftr = get_pixel_feature_v2(in_img, posx, posy, self.pixel_feature_neighbor_half_size)
                
                
                if 0:
                    if self.ori_img_precomputed_context_ftr_dir:
    #                     print 'use precomputed pixel context feature'
                        precomputed_context_file_path = os.path.join\
                        (self.ori_img_precomputed_context_ftr_dir, self.tr_in_imgs[k][:-4] + '_context_ftr.mat')
                        precomputed_context_file = scipy.io.loadmat(precomputed_context_file_path)
                        precomputed_context_ftr = precomputed_context_file['context_ftr']
                        precomputed_context_ftr = precomputed_context_ftr.reshape((h, w, self.local_context_paras['ftr_dim']))
                        pix_local_context = precomputed_context_ftr[posy, posx, :]
                    else:
    #                     print 'compute pixel context feature on the fly'
                        if preload_context_maps:
                            context_map = tr_context_maps[k]
                        else:       
                            in_img_context_path = os.path.join(self.sem_integral_map_dir, self.trSemIntegralMapFiles[k])
                            context_map = scipy.io.loadmat(in_img_context_path)
                            context_map = context_map['maps']
                            context_map = context_map.reshape((self.local_context_paras['label_num']))
                        assert context_map[0].shape[0] == (h + 100 * 2)
                        assert context_map[0].shape[1] == (w + 100 * 2)
                        pix_local_context = np.zeros\
                        ((posx.shape[0], self.local_context_paras['ftr_dim']), dtype=np.single)                 
                        getPixLocContextV2\
                        ([posx + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                          posy + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                          context_map, self.local_context_paras, pool, pix_local_context, self.num_proc])
#                     pix_local_context_2= pix_local_context.reshape((posx.shape[0],25,20))                  
#                     print 'pix_local_context2', np.sum(pix_local_context_2,axis=2)[:3,:]
                
                batch_pix_ftr_view[pc:pc + len(posx), :l_pix_ftr.shape[1]] = l_pix_ftr
                batch_in_pix_gradmag[0, pc:pc + len(posx)] = grad_mags_in_img[k][start:end]
                batch_enh_pix_gradmag[0, pc:pc + len(posx)] = grad_mags_enh_img[k][start:end]
                if 1:
                    batch_pix_local_context_view[pc:pc + len(posx), :] = pix_local_contexts[k][start:end, :]
                else:
                    batch_pix_local_context_view[pc:pc + len(posx), :] = pix_local_context
                batch_pix_to_imgID[pc:pc + len(posx)] = self.tr_img_id[k]
                pc += len(posx)
            pix_ftr_mean += np.sum(batch_pix_ftr, axis=1)
            batch_pix_local_context_mean += np.sum(batch_pix_local_context, axis=1)
            
            rand_perm = range(this_batch_size)
            rd.shuffle(rand_perm)
            batch_pix_ftr = batch_pix_ftr[:, rand_perm]
            batch_in_pix_gradmag = batch_in_pix_gradmag[:, rand_perm]
            batch_enh_pix_gradmag = batch_enh_pix_gradmag[:, rand_perm]
            batch_pix_local_context = batch_pix_local_context[:, rand_perm]
            batch_pix_to_imgID = batch_pix_to_imgID[rand_perm]
            
            l_batch = {}
            l_batch['data'] = batch_pix_ftr
            l_batch['in_pix_gradmag'] = batch_in_pix_gradmag
            l_batch['labels'] = batch_enh_pix_gradmag
            l_batch['pix_local_context'] = batch_pix_local_context
            l_batch['pix_to_imageID'] = batch_pix_to_imgID
            
            pickle(os.path.join(self.data_save_path_edge, 'data_batch_' + str(i + 1)), l_batch)
        print 'finish writing edge pixel batches'
        elapsed = time.time() - st_time
        print 'elapsed time: %f' % elapsed
        if self.preload_all_ori_img:
            del in_imgs
        
        print 'write edge pixel batches meta'
        pix_ftr_mean /= num_all_pixs
        batch_pix_local_context_mean /= num_all_pixs
        
        meta_dict = {}
        meta_dict['in_img_dir'] = self.ori_img_folder
        meta_dict['in_img_context_dir'] = self.sem_integral_map_dir
        meta_dict['enh_img_dir'] = self.enh_img_folder
        meta_dict['trSemIntegralMapFiles'] = self.trSemIntegralMapFiles
        meta_dict['tsSemIntegralMapFiles'] = self.tsSemIntegralMapFiles
        meta_dict['tr_img_id'] = self.tr_img_id
        meta_dict['ts_img_id'] = self.ts_img_id
        meta_dict['tr_in_imgs'] = self.tr_in_imgs
        meta_dict['tr_enh_imgs'] = self.tr_enh_imgs
        meta_dict['ts_in_imgs'] = self.ts_in_imgs
        meta_dict['ts_enh_imgs'] = self.ts_enh_imgs
        meta_dict['edge_pixs_x'] = edge_pixs_x
        meta_dict['edge_pixs_y'] = edge_pixs_y
        meta_dict['parameters'] = paras
        meta_dict['local_context_paras'] = self.local_context_paras

        meta_dict['data_mean'] = pix_ftr_mean
        meta_dict['pix_local_context_mean'] = batch_pix_local_context_mean
        meta_dict['imgCnnFtr'] = self.imgCnnFtr
        meta_dict['num_vis'] = pix_ftr_dim
        meta_dict['num_colors'] = 3
        meta_dict['img_size'] = 1
        
        pickle(os.path.join(self.data_save_path_edge, 'batches.meta'), meta_dict)        
        print 'finish writing edge pixel batches meta'

        if self.num_proc > 1:
            pool.close()           
    ''' write batch files for training a NN to regress global L-channel transform 
        (PCA components of control points of 2D transform curve given image global feature as input '''
    def write_global_batch_files(self):
        num_imgs = len(self.tr_in_imgs)
        if self.num_proc > 1:
            pool = mtp.Pool(processes=self.num_proc)
        paras = {}
        paras['in_img_dir'] = self.ori_img_folder
        paras['enh_img_dir'] = self.enh_img_folder
        paras['num_control_points'] = 21
        paras['fredo_image_processing'] = self.fredo_image_processing
        
        func_args = zip(self.tr_in_imgs, self.tr_enh_imgs, \
                      [paras] * num_imgs)
        st = time.time()
        if self.num_proc > 1:
            try:
                ctlPts = pool.map(compute_img_global_L_transform, func_args)
                ctlPts = ctlPts[0]
            except PatchPreparerError, e:
                print e
            
        else:
            ctlPts = []
            for i in range(num_imgs):
                if i % 50 == 0:
                    print '%d th out of %d images %s' % \
                    (i, num_imgs, self.tr_in_imgs[i])
                l_cp = compute_img_global_L_transform(func_args[i])
                ctlPts += [l_cp]    
        elapsed = time.time() - st        
        print 'compute_img_global_L_transform elapsed: %f' % elapsed
        print ctlPts[0].shape
        sys.exit()
        
        ctlPts = np.vstack(ctlPts)
        pca_solver = PCA(ctlPts)
        pca_mean, pca_evals, pca_evecs, pca_cum_frac_evals = \
        pca_solver.get_mean_evals_evecs_cumFracEval()
        print 'pca_cum_frac_evals', pca_cum_frac_evals
        # pcaCoeff: n * 1
        pcaCoeff = pca_solver.proj_topk(ctlPts, 1)
        pcaCoeff = pcaCoeff.reshape((1, num_imgs))
        
        pix_global_ftr = unpickle(os.path.join(self.ori_img_folder, 'ori_img_folder'))
        img_global_ftr_l2 = pix_global_ftr['img_global_ftr_l2']  # shape: n * d
        
        num_batches_global = 10
        batch_size = get_batch_size(num_imgs, num_batches_global)
        img_perm = range(num_imgs)
        rd.shuffle(img_perm)
        p = 0
        for i in range(num_batches_global):
            l_batch = {}
            l_batch['data'] = img_global_ftr_l2[p:p + batch_size[i], :]
            l_batch['labels'] = pcaCoeff[:, p:p + batch_size[i]]
            p += batch_size[i]
        meta = {}
        meta['data_mean'] = 0
        meta['num_vis'] = img_global_ftr_l2.shape[1]
                    
        if self.num_proc > 1:
            pool.close()
        
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
        num_img = len(self.in_imgs)
        imgsPath = [os.path.join(self.ori_img_folder, f) for f in self.in_imgs]
        segMapsPath = [os.path.join(self.semantic_map_dir, f) for f in self.semMapFiles]
        
        funcArgs = zip(imgsPath, segMapsPath, [self.img_color_integral_dir] * num_img)
        self.pool.map(compute_color_integral_map_helper, funcArgs)
        elapsed = time.time() - st
        print 'computing color integral maps is completed. %4.3f seconds' % elapsed
        return
#         apdWid = LOCAL_CONTEXT_MAP_APPEND_WIDTH
#         
#         for i in range(num_img):
#             st_time = time.time()
#             print 'process %d out of %d images %s' % (i, num_img, self.in_imgs[i])
#             in_img_path = os.path.join(self.ori_img_folder, self.in_imgs[i])            
#             if self.fredo_image_processing:
#                 in_img = read_tiff_16bit_img_into_LAB(in_img_path, 1.5)
#             else:
#                 in_img = read_tiff_16bit_img_into_LAB(in_img_path)
#             h, w, ch = in_img.shape[0], in_img.shape[1], in_img.shape[2]
#             assert ch == 3
#             
#             apdImg = np.zeros((h + 2 * apdWid, w + 2 * apdWid, 3), dtype=np.single)
#             apdImg[apdWid:apdWid + h, apdWid:apdWid + w, :] = in_img[:, :, :]   
#                      
#             if 1:
#                 print 'compute label-wise color integral map'
#                 semMap = scipy.io.loadmat(os.path.join(self.semantic_map_dir, self.semMapFiles[i]))
#                 semMap = semMap['responseMap']
#                 assert semMap.shape[:2] == in_img.shape[:2]
#                 apdSemMap = -np.ones((h + 2 * apdWid, w + 2 * apdWid), dtype=np.single)
#                 apdSemMap[apdWid:apdWid + h, apdWid:apdWid + w] = semMap
#                 
#                 colorIntegralMap = [None] * SEMANTIC_LABELS_NUM
#                 pixNumIntegralMap = [None] * SEMANTIC_LABELS_NUM
#                 
#                 colorIntegralMap = np.zeros((SEMANTIC_LABELS_NUM, h + 2 * apdWid, w + 2 * apdWid, 3), dtype=np.single)
#                 pixNumIntegralMap = np.zeros((SEMANTIC_LABELS_NUM, h + 2 * apdWid, w + 2 * apdWid), dtype=np.int32)
#                 
#                 for r in range(apdWid, 2 * apdWid + h):
#                     for c in range(apdWid, 2 * apdWid + w):
#                         sem = apdSemMap[r, c]
#                         colorIntegralMap[:, r, c, :] = colorIntegralMap[:, r - 1, c, :] + colorIntegralMap[:, r, c - 1, :] - \
#                         colorIntegralMap[:, r - 1, c - 1, :] 
#                         pixNumIntegralMap[:, r, c] = pixNumIntegralMap[:, r - 1, c] + pixNumIntegralMap[:, r, c - 1] - \
#                         pixNumIntegralMap[:, r - 1, c - 1]                      
#                         
#                         if sem >= 0:
#                             colorIntegralMap[sem, r, c, :] += apdImg[r, c, :]
#                             pixNumIntegralMap[sem, r, c] += 1
#                                                     
#                 mat_dict = {}
#                 mat_dict['colorIntegralMaps'] = colorIntegralMap
#                 mat_dict['pixNumIntegralMaps'] = pixNumIntegralMap                
#             else:
#                 colorIntegralMap = np.zeros((h + 2 * apdWid, w + 2 * apdWid, 3), dtype=np.single)
#                 # fill up integral map in scanning-line order
#                 for r in range(apdWid, 2 * apdWid + h):
#                     for c in range(apdWid, 2 * apdWid + w):
#                         colorIntegralMap[r, c, :] = colorIntegralMap[r - 1, c, :] + colorIntegralMap[r, c - 1, :] - \
#                         colorIntegralMap[r - 1, c - 1, :] + apdImg[r, c]
#                 integral_map_2 = colorIntegralMap.swapaxes(0, 1).swapaxes(0, 2)  # shape: (3,h,w)  
#                 mat_dict = {}
#                 mat_dict['colorIntegralMaps'] = integral_map_2
#             mat_path = os.path.join(self.img_color_integral_dir, self.in_imgs[i][:-4] + '.mat')
#             scipy.io.savemat(mat_path, mat_dict)
#             elapsed = time.time() - st_time
#             print 'elapsed time', elapsed
                    
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
            print 'load context map: %s' % colorIntegralFileNm
            context_map = scipy.io.loadmat(colorIntegralFileNm)
            context_map = context_map['colorIntegralMaps']
            # context_map shape: (3,h,w)         
            assert h == (context_map.shape[1] - 2 * apdWid)
            assert w == (context_map.shape[2] - 2 * apdWid)
            print 'h,w', h, w
            context_map = context_map.reshape\
            ((3, (h + 2 * apdWid), (w + 2 * apdWid)))  # L,a,b three channels
            
            context_ftrs = np.zeros((h * w, self.local_context_paras['pool_region_num'] * 3), \
                                    dtype=np.single)
            pix_x, pix_y = np.meshgrid(range(w), range(h))
            pix_x, pix_y = pix_x.flatten(), pix_y.flatten()
            st_time = time.time()
            get_pixel_local_context_mean_color\
            ([pix_x + apdWid, pix_y + apdWid, context_map, self.local_context_paras, pool, context_ftrs])            
            elapsed = time.time() - st_time
            print 'elapsed: %f' % (elapsed)
            
            context_ftrs = context_ftrs.reshape((h, w, self.local_context_paras['pool_region_num'] * 3))
            mat_dict = {}
            mat_dict['context_mean_color_ftr'] = context_ftrs
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
    #            coxtFileNm = os.path.join(self.sem_integral_map_dir,\
    #                                          self.semIntegralMapFiles[i])            
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
        op.add_option("ori-img-saliency-folder", "ori_img_saliency_folder", StringOptionParser, \
                      "original image saliency map folder", default="")  
        op.add_option("enh-img-folder", "enh_img_folder", StringOptionParser, \
                      "enhanced image folder", default="")
        op.add_option("img-Cnn-ftr-file", "img_Cnn_ftr_file", StringOptionParser, "the path of image CNN feature file", \
                      default='')
        op.add_option("data-save-path", "data_save_path", StringOptionParser, "path of saving folder", default="")
        op.add_option("data-save-path-edge", "data_save_path_edge", StringOptionParser, "path of saving folder for edge pixels", default="")
        
        op.add_option("patch-data", "patch_data", BooleanOptionParser,
                      "store patch data?", default=False)                              
        op.add_option("pixel-feature-neighbor-half-size", "pixel_feature_neighbor_half_size", IntegerOptionParser,
                      "half size of neighborhood to estimate pixel feature", default=3)
        # patch side length would be 2*x+1
        op.add_option("cnn-patch-half-size", "cnn_patch_half_size", IntegerOptionParser, \
                      "half size of patch in CNN ", default=16)
        # # neighborhood side length would be 2*x+1
        op.add_option("color-mapping-neighbor-size", 'color_mapping_neighbor_size', IntegerOptionParser, \
                      "size of neighborhood in color mapping estimation", default=2)
        op.add_option("color-mapping-residue-thres", "color_mapping_residue_thres", FloatOptionParser, "threshold of mean residue ||r||_2^2 in color mapping estimation", default=4.0)
        op.add_option("stride", "stride", IntegerOptionParser, "stride", default=OptionExpression("cnn_patch_half_size"))
        op.add_option("ls-lambda", "ls_lambda", FloatOptionParser, \
                      "coefficient of the 2nd term in regularized least square regression", default=1e-1)
        op.add_option("ls-atol", "ls_atol", FloatOptionParser, "tolerance coefficient a", default=1e-6)
        op.add_option("outlier-label-thres", "outlier_label_thres", FloatOptionParser, \
                       "threshold of multiple of standard deviation to detect outlier labels", default=6.0)
             
        op.add_option("color-mapping-PCA-thres", "color_mapping_PCA_thres", FloatOptionParser, \
                      "threshold of cumulative eigenvalues", default=.9)
        op.add_option("num-batches", 'num_batches', IntegerOptionParser, "number of batches", default=100)
        op.add_option("num-edge-batches", 'num_edge_batches', IntegerOptionParser,
                      "number of edge pixel batches", default=50)        
        op.add_option("preload-all-ori-img", 'preload_all_ori_img', BooleanOptionParser,
                      "enable it if memory is large enough to accommodate all original images", default=True)
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
        op.add_option("do-precompute-local-context-mean-color-ftr", 'do_precompute_local_context_mean_color_ftr', BooleanOptionParser,
                      "precompute image local context mean color feature", default=0)                 
        op.add_option("do-local-context-SPM-regions", 'do_local_context_SPM_regions', BooleanOptionParser,
                      "add 3 additional Sptial Pyramid Matching pooling regions to context feature", default=0)          
        
        op.add_option("do-precompute-local-context-ftr", 'do_precompute_local_context_ftr', BooleanOptionParser,
                      "precompute local context ftr?", default=0, requires=['ori_img_precomputed_context_ftr_dir'])         
        op.add_option("do-write-color-batch-files", 'do_write_color_batch_files', BooleanOptionParser,
                      "write color batch files?", default=0) 
        op.add_option("do-write-edge-pix-batch-files", 'do_write_edge_pix_batch_files', BooleanOptionParser,
                      "write edge pix batch files?", default=0)

        op.add_option("is-first-effect", 'is_first_effect', BooleanOptionParser,
                      "Am I training first effect?", default=0)  
        op.add_option("sem-fgbg-matfile", 'sem_fgbg_matfile', StringOptionParser,
                      "mat file for mapping semantic to binary foreground/background boolean?", default='')                               
        op.add_option("context-ftr-baseline", 'context_ftr_baseline', BooleanOptionParser,
                      "baseline implementation of context feature", default=0)                               
                
        return op
    
        
if __name__ == "__main__":

    op = TrainPatchPreparer.get_options_parser()
    op = TrainPatchPreparer.parse_options(op)
    preparer = TrainPatchPreparer(op)
    preparer.find_images()
    preparer.find_segments()
    preparer.findSemanticMaps()
    preparer.findSemIntegralFiles()
    preparer.load_image_feature()
    preparer.compute_image_global_ftr()
    
    preparer.compute_color_integral_map()        
    preparer.findColorIntegralFiles()
##     preparer.precompute_local_context_color_ftr()
##    preparer.precompute_local_context_ftr()
    
    preparer.write_color_batch_files()
    ## preparer.write_edge_pix_batch_files()
##     preparer.write_global_batch_files()
    preparer.clear()
