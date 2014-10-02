import sys
#sys.path.append(r'/home/zyan3/proj/cuda-convnet-plus/cuda-convnet-plus/py')
#sys.path.append(r'D:\proj\cuda-convnet-plus\cuda-convnet-plus\py')
#sys.path.append(r'D:\yzc\proj\cuda-convnet-plus\cuda-convnet-plus\py')
sys.path.append(r'/home/yzc/yzc/proj/cuda-convnet-plus/cuda-convnet-plus/py')

import os
import numpy as np
import numpy.random as rd
import time
import math
import fnmatch
from subprocess import call
import copy

from util import *
from util_image import *
from colorMapping import *
from PrinCompAnal import *
import scipy.misc
import scipy.ndimage
import scipy.io
import scipy.cluster
import matplotlib.pyplot as mpplot
from pylab import *

LOCAL_CONTEXT_MAP_APPEND_WIDTH = 100
SEMANTIC_LABELS_NUM = 20

class UtilCnnImageEnhanceError(Exception):
    pass


    # assume saliency_map value range [0,1]
def get_samples_pos(h, w, patch_hs, stride, saliency_map=None):
    samplesY, samplesX = np.int(floor((h - 2 * patch_hs - 1) / stride) + 1), np.int(floor((w - 2 * patch_hs - 1) / stride) + 1)
    samplesPosY = np.array([patch_hs + i * stride for i in range(samplesY)])
    samplesPosX = np.array([patch_hs + i * stride for i in range(samplesX)])
    posX, posY = np.meshgrid(samplesPosX, samplesPosY)
    
    if saliency_map is not None:
        print 'use saliency map to sample patches'
        posX, posY = posX.flatten(), posY.flatten()
#         print 'before saliency filtering, len posY : %d' % len(posY)
        pos_saliency = saliency_map[posY, posX]
#         print 'len pos_saliency:%d' % len(pos_saliency) 
        rand_val = rd.random_sample(pos_saliency.shape)
        sample_mask = [pair[0] > pair[1] for pair in zip(pos_saliency, rand_val)]
        sample_idx = np.nonzero(sample_mask)[0]
        posX, posY = posX[sample_idx], posY[sample_idx]
    return posX, posY

def get_patches(img, posx, posy, patch_hs, patches=None):
    # img: h*w*ch 
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    st_time = time.time()
    
    ch = img.shape[2]
    numPos = len(posx)
    side = 2 * patch_hs + 1
    if patches == None:
        patches = np.zeros((numPos, side, side, ch), dtype=img.dtype)
#     else:
#         print 'get_patches patches are not none'
#         print 'numPos,patches shape',numPos,patches.shape
        
#     print 'h,w',h,w
    for i in range(numPos):
        cx, cy = posx[i], posy[i]
#         print 'cx,cy', cx, cy
        p_start_x = 0 if cx >= patch_hs else patch_hs - cx
        p_end_x = side if cx + patch_hs < w else w - cx + patch_hs
        p_start_y = 0 if cy >= patch_hs else patch_hs - cy
        p_end_y = side if cy + patch_hs < h else h - cy + patch_hs

        img_start_x = cx - patch_hs if cx >= patch_hs else 0
        img_end_x = cx + patch_hs + 1 if cx + patch_hs < w else w
        img_start_y = cy - patch_hs if cy >= patch_hs else 0
        img_end_y = cy + patch_hs + 1 if cy + patch_hs < h else h
        
#         print 'p start end',p_start_x,p_end_x,p_start_y,p_end_y
#         print 'img start end', img_start_x,img_end_x,img_start_y,img_end_y

        patches[i, p_start_y:p_end_y, p_start_x:p_end_x, :] = img[img_start_y:img_end_y, img_start_x:img_end_x, :]
    elapsed_time = time.time() - st_time
#     print 'get_patches elapsed_time: %f ' % elapsed_time
    return patches

def get_patches_v2(expanded_img, patches, posx, posy, patch_hs):
    st_time = time.time()
#     h, w, c = img.shape[0], img.shape[1], img.shape[2]
#     print 'h,w',h,w
    side = 2 * patch_hs + 1
    assert posx.shape[0] == posy.shape[0]
    for i in range(posx.shape[0]):
#         print 'sy ey sx ex',posy[i]+patch_hs,posy[i]+patch_hs+side,posx[i]+patch_hs,posx[i]+patch_hs+side
        patches[i, :, :, :] = expanded_img[posy[i]:posy[i] + side, \
                                      posx[i]:posx[i] + side, :]
    elapsed_time = time.time() - st_time
#     print 'get_patches_v2 elapsed_time:%f' % elapsed_time
    

# sample patches centered at pixels in certain rows
def get_patches_for_image_rows(img, expanded_img, patches, start_row, end_row, patch_hs):
    st_time = time.time()
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    side = 2 * patch_hs + 1
    
#     patches = np.zeros((side, side, end_row - start_row, w, c), dtype=img.dtype)
#     patches = np.zeros((end_row - start_row, w, side, side, c), dtype=img.dtype)

    if 1:
    # a quick hack here for speed up. To be fixed if local context feature is introduced        
        patches[patch_hs, patch_hs, :, :, :] = expanded_img[start_row + patch_hs:end_row + patch_hs, patch_hs:patch_hs + w, :]
    else:
        for dy in range(-patch_hs, patch_hs + 1):
            for dx in range(-patch_hs, patch_hs):
    #             patches[:,:,dy + patch_hs, dx + patch_hs,:] = \
                patches[dy + patch_hs, dx + patch_hs, :, :, :] = \
                 expanded_img[start_row + patch_hs + dy:end_row + patch_hs + dy, patch_hs + dx:patch_hs + w + dx, :]            
    elapsed_1 = time.time() - st_time
#     patches = patches.reshape(((end_row-start_row)*w,side,side,c))
#     return patches.swapaxes(0,3)
    st_time = time.time()
#     patches2 = patches.swapaxes(0, 2).swapaxes(1, 3)
#     patches2 = patches2.reshape((w * (end_row - start_row), side, side, c))
    elapsed_2 = time.time() - st_time
#     print 'elapsed_1,elapsed_2',elapsed_1,elapsed_2
#     sys.exit()
#     return patches2

# get pixel data of patchese centereed at each pixel in a region of image 
def get_patches_for_image_region(img, top, left, bottom, right, patch_hs):
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    assert top >= patch_hs and left >= patch_hs
    assert bottom < h - patch_hs and right < w - patch_hs
    
    st_time = time.time()
    
    side = 2 * patch_hs + 1
    
    rg_h, rg_w = bottom - top, right - left
    
    patches = np.zeros((side, side, rg_h, rg_w, ch), dtype=img.dtype)
    for dy in range(-patch_hs, patch_hs + 1):
        for dx in range(-patch_hs, patch_hs + 1):
            patches[dy + patch_hs, dx + patch_hs, :, :, :] = img[top + dy:top + dy + rg_h, left + dx:left + dx + rg_w, :]
    patches = patches.swapaxes(0, 2).swapaxes(1, 3)
    patches = patches.reshape((rg_h * rg_w, side, side, ch))
    
    elapsed_time = time.time() - st_time
    print 'elapsed_time: %f ' % elapsed_time
    
    return patches

def load_segment(seg_file_path, img_shape=None):
    f = open(seg_file_path, 'r')
    num_comp = int(f.readline())
    print 'load segmentation %s. %d components' % (seg_file_path, num_comp)
    seg = [None] * num_comp
    for j in range(num_comp):
        l_comp = f.readline().split(' ')
        size_comp = int(l_comp[0])
        pxs = [int(idx) for idx in l_comp[1:]]
        assert size_comp == len(pxs)
        seg[j] = pxs
    f.close()
#     print 'seg[0]',seg[0]
    # validate segment
    if img_shape:
        id_max = img_shape[0] * img_shape[1] - 1
        id_max2 = [max(comp) for comp in seg]
        if not max(id_max2) == id_max:
	    print img_shape,seg_file_path
            raise UtilCnnImageEnhanceError\
                ("invalid max idx in segment. expected:%d in fact:%d" % (id_max, max(id_max2)))
    return seg  

def get_local_context_color(img_ab, px, py, local_context_color_paras, res=None):
    h, w, ch = img_ab.shape[0], img_ab.shape[1], img_ab.shape[2]
    assert ch == 2
    hf_sz, hist_bin_num = \
    local_context_color_paras['hf_sz'], local_context_color_paras['hist_bin_num']
    xmin = px - hf_sz if (px - hf_sz) >= 0 else 0
    xmax = px + hf_sz + 1 if (px + hf_sz + 1) < w else w
    ymin = py - hf_sz if (py - hf_sz) >= 0 else 0
    ymax = py + hf_sz + 1 if (py + hf_sz + 1) < h else h
    region = img_ab[ymin:ymax, xmin:xmax, :]
#     print 'region shape',region.shape
    h2, w2 = ymax - ymin, xmax - xmin
    region = region.reshape((h2 * w2, 2))
    
#     print 'region shape',region.shape
    hist_a, bin_edges_a = np.histogram(region[:, 0], hist_bin_num, range=(-128, 128))
    hist_b, bin_edges_b = np.histogram(region[:, 1], hist_bin_num, range=(-128, 128))
    hist_a = np.single(hist_a) / np.single(h2 * w2)
    hist_b = np.single(hist_b) / np.single(h2 * w2)
#     print 'hist,hist_sum',hist,np.sum(hist)
    if res == None:
        return np.concatenate((hist_a, hist_b))
    else:
        res[:hist_bin_num] = hist_a
        res[hist_bin_num:(2 * hist_bin_num)] = hist_b

# def get_local_context_color(img_ab,px,py,local_context_color_paras):
#     h,w=img_ab.shape[0],img_ab.shape[1]
#     hf_sz,hist_bin_num=\
#     local_context_color_paras['hf_sz'],local_context_color_paras['hist_bin_num']
#     xmin = px-hf_sz if (px-hf_sz)>=0 else 0
#     xmax = px+hf_sz+1 if (px+hf_sz+1)<w else w
#     ymin = py-hf_sz if (py-hf_sz)>=0 else 0
#     ymax=py+hf_sz+1 if (py+hf_sz+1)<h else h
#     region = img_ab[ymin:ymax,xmin:xmax]
# #     print 'region shape',region.shape
#     region=region.flatten()
# #     print 'region shape',region.shape
#     hist,bin_edges=np.histogram(region,hist_bin_num,range=(-128,128))
#     hist = np.single(hist) / np.single((ymax-ymin)*(xmax-xmin))
# #     print 'hist,hist_sum',hist,np.sum(hist)
#     return hist

def compute_color_integral_map_helper(inArgs):
    imgPath, semMapPath, img_color_integral_dir= inArgs[0],inArgs[1],inArgs[2]
    imgDir,imgNm = os.path.split(imgPath)
    in_img = read_tiff_16bit_img_into_LAB(imgPath)
    h, w, ch = in_img.shape[0], in_img.shape[1], in_img.shape[2]
    assert ch == 3
    
    apdWid=LOCAL_CONTEXT_MAP_APPEND_WIDTH
    apdImg = np.zeros((h + 2 * apdWid, w + 2 * apdWid, 3), dtype=np.single)
    apdImg[apdWid:apdWid + h, apdWid:apdWid + w, :] = in_img[:, :, :]   

    mat_path = os.path.join(img_color_integral_dir, imgNm[:-4] + '.mat')
    if os.path.exists(mat_path):
        print 'mat file %s already exists' % mat_path
        return
             
    if 1:
        print 'compute label-wise color integral map'
        semMap = scipy.io.loadmat(semMapPath)
        semMap = semMap['responseMap']
        assert semMap.shape[:2] == in_img.shape[:2]
        apdSemMap = -np.ones((h + 2 * apdWid, w + 2 * apdWid), dtype=np.single)
        apdSemMap[apdWid:apdWid + h, apdWid:apdWid + w] = semMap
        
        colorIntegralMap = [None] * SEMANTIC_LABELS_NUM
        pixNumIntegralMap = [None] * SEMANTIC_LABELS_NUM
        
        colorIntegralMap = np.zeros((SEMANTIC_LABELS_NUM, h + 2 * apdWid, w + 2 * apdWid, 3), dtype=np.single)
        pixNumIntegralMap = np.zeros((SEMANTIC_LABELS_NUM, h + 2 * apdWid, w + 2 * apdWid), dtype=np.int32)
        
        for r in range(apdWid, 2 * apdWid + h):
            for c in range(apdWid, 2 * apdWid + w):
                sem = apdSemMap[r, c]
                colorIntegralMap[:, r, c, :] = colorIntegralMap[:, r - 1, c, :] + colorIntegralMap[:, r, c - 1, :] - \
                colorIntegralMap[:, r - 1, c - 1, :] 
                pixNumIntegralMap[:, r, c] = pixNumIntegralMap[:, r - 1, c] + pixNumIntegralMap[:, r, c - 1] - \
                pixNumIntegralMap[:, r - 1, c - 1]                      
                
                if sem>=0:
                    colorIntegralMap[sem, r, c, :] += apdImg[r, c, :]
                    pixNumIntegralMap[sem, r, c] += 1
                                            
        mat_dict = {}
        mat_dict['colorIntegralMaps'] = colorIntegralMap
        mat_dict['pixNumIntegralMaps'] = pixNumIntegralMap                
    else:
        colorIntegralMap = np.zeros((h + 2 * apdWid, w + 2 * apdWid, 3), dtype=np.single)
        # fill up integral map in scanning-line order
        for r in range(apdWid, 2 * apdWid + h):
            for c in range(apdWid, 2 * apdWid + w):
                colorIntegralMap[r, c, :] = colorIntegralMap[r - 1, c, :] + colorIntegralMap[r, c - 1, :] - \
                colorIntegralMap[r - 1, c - 1, :] + apdImg[r, c]
        integral_map_2 = colorIntegralMap.swapaxes(0, 1).swapaxes(0, 2)  # shape: (3,h,w)  
        mat_dict = {}
        mat_dict['colorIntegralMaps'] = integral_map_2
    scipy.io.savemat(mat_path, mat_dict)  


def processImgs(in_args):
    imgFileNm, enhImgFileNm, imgSegFileNm, semMapFileNm, semIntegralMapFileNm, colorIntegralFileNm, \
    paras = in_args[0], in_args[1], in_args[2], in_args[3], in_args[4], in_args[5], in_args[6]
     
    local_context_color = False if colorIntegralFileNm == None else True
    print local_context_color
    in_img_dir, enh_img_dir, in_seg_dir, \
    sem_map_dir,sem_integral_map_dir, \
    img_color_integral_dir, ori_img_edge_folder, patch_half_size, \
    nbSize, stride, \
    segment_random_sample_num, local_context_paras, fredo_image_processing = \
    paras['in_img_dir'], paras['enh_img_dir'], paras['in_seg_dir'], \
    paras['semantic_map_dir'], paras['sem_integral_map_dir'], \
    paras['img_color_integral_dir'], paras['ori_img_edge_folder'], paras['patch_half_size'], \
    paras['nbSize'], paras['stride'], \
    paras['segment_random_sample_num'], paras['local_context_paras'], paras['fredo_image_processing']
    if local_context_color:
        local_context_color_paras = paras['local_context_color_paras']
#     print patch_half_size,nbSize,stride, meanResidueThres,gauSigma,lsLambda,lsAtol
    in_img_path, enh_img_path = \
    os.path.join(in_img_dir, imgFileNm), \
    os.path.join(enh_img_dir, enhImgFileNm)
    in_img_seg_path = os.path.join(in_seg_dir, imgSegFileNm)
    if fredo_image_processing == 1:
        in_img, enh_img = read_tiff_16bit_img_into_LAB(in_img_path, 1.5, False), \
        read_tiff_16bit_img_into_LAB(enh_img_path, 0, False)        
    else:
        in_img, enh_img = read_tiff_16bit_img_into_LAB(in_img_path), \
        read_tiff_16bit_img_into_LAB(enh_img_path)
    h, w, c = in_img.shape[0], in_img.shape[1], in_img.shape[2]        
    
    print semMapFileNm
    semMap = scipy.io.loadmat(os.path.join(sem_map_dir, semMapFileNm))
    semMap = semMap['responseMap']
    if not semMap.shape == in_img.shape[:2]:
        print 'semMap shape',semMap.shape
        print 'in_img shape',in_img.shape[:2]
    assert semMap.shape == in_img.shape[:2]
    
    semIntegralMap = scipy.io.loadmat(os.path.join(sem_integral_map_dir, semIntegralMapFileNm))
    semIntegralMap = semIntegralMap['maps']
    semIntegralMap = semIntegralMap.reshape((local_context_paras['label_num']))
    assert (semIntegralMap[0].shape[0] == (h + 100 * 2)) & (semIntegralMap[0].shape[1] == (w + 100 * 2))

    if local_context_color:
        colorIntegralMaps_ = scipy.io.loadmat(os.path.join(img_color_integral_dir, colorIntegralFileNm))
        colorIntegralMaps = colorIntegralMaps_['colorIntegralMaps']
        pixNumIntegralMaps = colorIntegralMaps_['pixNumIntegralMaps']
        assert (colorIntegralMaps.shape[1] == (h + 100 * 2)) & (colorIntegralMaps.shape[2] == (w + 100 * 2))

    if not in_img.shape[2] == 3:
        raise UtilCnnImageEnhanceError('Error: not a 3-channel image. %s' % in_img_path)
    if not in_img.shape == enh_img.shape:
        print 'in_img shape, enh_img shape', in_img.shape, enh_img.shape
        raise UtilCnnImageEnhanceError('Error: in_img has a different shape from enh_img. %s' % in_img_path)

    in_img_seg = load_segment(in_img_seg_path, in_img.shape)
    num_seg = len(in_img_seg)

    segPosx, segPosy = np.zeros((num_seg), dtype=np.int32), np.zeros((num_seg), dtype=np.int32)
    for i in range(num_seg):
        px_x, px_y = np.array(in_img_seg[i]) % w, np.array(in_img_seg[i]) / w
        segPosx[i] = np.round(np.mean(px_x))
        segPosy[i] = np.round(np.mean(px_y))

    pixContextSemFtr = getPixContextSem\
    ([segPosx + LOCAL_CONTEXT_MAP_APPEND_WIDTH, segPosy + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
      semIntegralMap, local_context_paras, None, None])
    if local_context_color:
        pixContextColorFtr, pixContextPixnumFtr = getPixContextColorFtr\
        (segPosx + LOCAL_CONTEXT_MAP_APPEND_WIDTH, segPosy + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
          semMap[segPosy,segPosx], colorIntegralMaps, pixNumIntegralMaps,local_context_paras)        
    
    
    
#     precomputer_context_mean_color_mat_nm = os.path.join\
#     (in_img_precomputed_context_mean_color_dir, imgFileNm[:-4] + '_context_mean_color_ftr.mat')
#     precomputer_context_mean_color = scipy.io.loadmat(precomputer_context_mean_color_mat_nm)
#     precomputer_context_mean_color = precomputer_context_mean_color['context_mean_color_ftr']
#     pix_context_mean_color_ftr = precomputer_context_mean_color[segPosy, segPosx, :]

    if segment_random_sample_num > 1:
        # randomly sample a fixed number of pixels in segment
        in_img = in_img.reshape((h * w, c))
        enh_img = enh_img.reshape((h * w, c))
    if local_context_color:
        in_img_3d = in_img.reshape((h, w, c))
        

    
    seg_in_pixs = np.zeros((num_seg, segment_random_sample_num, 3), dtype=np.single)
    seg_enh_pixs = np.zeros((num_seg, segment_random_sample_num, 3), dtype=np.single)
    
    if local_context_color:
        seg_local_context_color = np.zeros((num_seg, 2 * local_context_color_paras['hist_bin_num']), dtype=np.single)
 

    for i in range(num_seg):
        seg = np.array(in_img_seg[i])
#         print 'seg',seg
        seg_sz = seg.shape[0]
#         print 'seg_sz',seg_sz
        mult = div_up(segment_random_sample_num, seg_sz)
#         print 'mult',mult
        px_x, px_y = seg % w, seg / w
        cp_x, cp_y = np.round(np.mean(px_x)), np.round(np.mean(px_y))
        if segment_random_sample_num == 1:
            seg_in_pixs[i, 0, :] = in_img[cp_y, cp_x, :]
            seg_enh_pixs[i, 0, :] = enh_img[cp_y, cp_x, :]
        else:
            permu = np.tile(np.array(range(seg_sz)), mult)
            rd.shuffle(permu)
    #         print 'permu',permu
    #         print seg[permu[:segment_random_sample_num]]
            seg_in_pixs[i, :, :] = in_img[seg[permu[:segment_random_sample_num]], :]
            seg_enh_pixs[i, :, :] = enh_img[seg[permu[:segment_random_sample_num]], :]
            
        if local_context_color:
            seg_local_context_color[i, :] = \
            get_local_context_color(in_img_3d[:, :, 1:3], cp_x, cp_y, local_context_color_paras)

    if np.min(segPosx) < 0 or np.max(segPosx) >= w or np.min(segPosy) < 0 or np.max(segPosy) >= h:
        raise UtilCnnImageEnhanceError('image %s\n segment centers is out of boundary.' % in_img_path)
    if local_context_color:
        return len(segPosx), segPosx, segPosy, seg_in_pixs, \
             seg_enh_pixs, pixContextSemFtr, pixContextColorFtr, pixContextPixnumFtr, seg_local_context_color
    else:
        return len(segPosx), segPosx, segPosy, seg_in_pixs, \
             seg_enh_pixs, pixContextSemFtr


def process_img_edge_pix(in_args):
    in_img_filenm, enh_img_filenm, in_img_context_filenm, paras = \
    in_args[0], in_args[1], in_args[2], in_args[3]   
    in_img_dir, enh_img_dir, in_img_context_dir, \
    ori_img_precomputed_context_ftr_dir, ori_img_edge_folder, \
    gaus_smooth_sigma, local_context_paras, \
    fredo_image_processing = \
    paras['in_img_dir'], paras['enh_img_dir'], paras['in_img_context_dir'], \
    paras['ori_img_precomputed_context_ftr_dir'], \
    paras['ori_img_edge_folder'], paras['gaus_smooth_sigma'], \
    paras['local_context_paras'], paras['fredo_image_processing']
    
    in_img_path, enh_img_path = \
    os.path.join(in_img_dir, in_img_filenm), \
    os.path.join(enh_img_dir, enh_img_filenm)    
    if fredo_image_processing == 1:
        in_img, enh_img = read_tiff_16bit_img_into_LAB(in_img_path, 1.5, False), \
        read_tiff_16bit_img_into_LAB(enh_img_path, 0, False)        
    else:
        in_img, enh_img = read_tiff_16bit_img_into_LAB(in_img_path), \
        read_tiff_16bit_img_into_LAB(enh_img_path)    
    h, w, c = in_img.shape[0], in_img.shape[1], in_img.shape[2]        
    
    in_img_context_path = os.path.join(in_img_context_dir, in_img_context_filenm)
    context_map = scipy.io.loadmat(in_img_context_path)
    context_map = context_map['maps']
    context_map = context_map.reshape((local_context_paras['label_num']))
    assert context_map[0].shape[0] == (h + 100 * 2)
    assert context_map[0].shape[1] == (w + 100 * 2)


    in_img_edge_mat_path = os.path.join(ori_img_edge_folder, in_img_filenm[:-4] + '_edge.mat')
    print 'load edge pixel mat file: %s' % in_img_edge_mat_path
    edge_pix = scipy.io.loadmat(in_img_edge_mat_path)
    edge_pix_y, edge_pix_x = np.int32(edge_pix['edge_pix_y'] - 1), \
    np.int32(edge_pix['edge_pix_x'] - 1)  # note, matlab uses 1-based index 
    num_edge_pix = edge_pix_y.shape[0]
    edge_pix_y, edge_pix_x = edge_pix_y.reshape((num_edge_pix)), edge_pix_x.reshape((num_edge_pix))
    # take more pixels in 5*5 local neighborhood
    extend_width = 1
    uni_edge_pix_x_extended, uni_edge_pix_y_extended = \
    get_extended_edge_pixel(edge_pix_x, edge_pix_y, h, w, extend_width)
    
#     edge_pix_y_extended, edge_pix_x_extended = np.array([]), np.array([])
#     for dy in range(-extend_width, extend_width + 1):
#         for dx in range(-extend_width, extend_width + 1):
#             l_edge_pix_y, l_edge_pix_x = edge_pix_y + dy, edge_pix_x + dx
#             idx = np.nonzero((l_edge_pix_y >= 0) & (l_edge_pix_y < h) & (l_edge_pix_x >= 0) & (l_edge_pix_x < w))
# #             print 'l_edge_pix_y[idx[0]] shape dtype', np.array(l_edge_pix_y[idx[0]]).shape, \
#             np.array(l_edge_pix_y[idx[0]]).dtype
#             edge_pix_y_extended = np.hstack((edge_pix_y_extended, l_edge_pix_y[idx[0]]))
#             edge_pix_x_extended = np.hstack((edge_pix_x_extended, l_edge_pix_x[idx[0]]))            
#     edge_pix_y_extended = np.int32(edge_pix_y_extended)
#     edge_pix_x_extended = np.int32(edge_pix_x_extended)
#     
# #     mask_img = np.zeros((h,w,c),dtype=np.uint8)
# #     mask_img[edge_pix_y_extended,edge_pix_x_extended,:]=255
# #     mpplot.figure()
# #     mpplot.imshow(mask_img)
# #     mpplot.show()
#     
#     # get unique (x,y) pairs
#     uni_xy = {}
#     for i in range(edge_pix_y_extended.shape[0]):
#         uni_xy[edge_pix_x_extended[i] + edge_pix_y_extended[i] * w] = True
# #     print 'unique xy pairs:%d' % len(uni_xy)
#     uni_xy_vals = np.array(uni_xy.keys())
#     uni_edge_pix_y_extended, uni_edge_pix_x_extended = \
#     np.array(uni_xy_vals / w), np.array(uni_xy_vals % w)
# 
# #     print 'edge_pix_y_extended shape dtype', edge_pix_y_extended.shape, edge_pix_y_extended.dtype
#     print 'edge_pix_x_extended shape dtype', edge_pix_x_extended.shape, edge_pix_x_extended.dtype
# #     print 'uni_edge_pix_y_extended shape dtype', uni_edge_pix_y_extended.shape, \
# #     uni_edge_pix_y_extended.dtype
#     print 'uni_edge_pix_x_extended shape dtype', uni_edge_pix_x_extended.shape, \
#     uni_edge_pix_x_extended.dtype
    # compute gradient (central difference) of input/enhanced image
    in_img_L, enh_img_L = in_img[:, :, 0], enh_img[:, :, 0]
    in_img_L_smooth = scipy.ndimage.filters.gaussian_filter(in_img_L, gaus_smooth_sigma)
    enh_img_L_smooth = scipy.ndimage.filters.gaussian_filter(enh_img_L, gaus_smooth_sigma)
    grady_in_img_L, gradx_in_img_L = np.gradient(in_img_L_smooth)
    grady_enh_img_L, gradx_enh_img_L = np.gradient(enh_img_L_smooth)
#     mpplot.figure()
#     mpplot.subplot(2, 2, 0)
#     mpplot.imshow(in_img_L, cmap=get_cmap('Greys'))
#     mpplot.title('in_img_L')
#     mpplot.subplot(2, 2, 1)
#     mpplot.imshow(in_img_L_smooth, cmap=get_cmap('Greys'))
#     mpplot.title('in_img_L_smooth')    
#     mpplot.subplot(2, 2, 2)
#     mpplot.imshow(enh_img_L, cmap=get_cmap('Greys'))
#     mpplot.title('enh_img_L')
#     mpplot.subplot(2, 2, 3)
#     mpplot.imshow(enh_img_L_smooth, cmap=get_cmap('Greys'))
#     mpplot.title('enh_img_L_smooth')
#     mpplot.show()
    
    grad_mag_in_img_L = np.sqrt(grady_in_img_L ** 2 + gradx_in_img_L ** 2)
    grad_mag_enh_img_L = np.sqrt(grady_enh_img_L ** 2 + gradx_enh_img_L ** 2)
#     print 'grad_mag_in_img_L', grad_mag_in_img_L[0, :40]
#     print 'grad_mag_enh_img_L', grad_mag_enh_img_L[0, :40]
    
    # need to make sure 'pix_local_context' won't consume too much main memory
    if ori_img_precomputed_context_ftr_dir:
        print 'use precomputed pixel context feature'
        precomputed_context_file_path = os.path.join\
        (ori_img_precomputed_context_ftr_dir, in_img_filenm[:-4] + '_context_ftr.mat')
        precomputed_context_file = scipy.io.loadmat(precomputed_context_file_path)
        precomputed_context_ftr = precomputed_context_file['context_ftr']
        precomputed_context_ftr = precomputed_context_ftr.reshape((h, w, local_context_paras['ftr_dim']))
        pix_local_context = precomputed_context_ftr[uni_edge_pix_y_extended, uni_edge_pix_x_extended, :]
    else:
        print 'computer pixel context feature on the fly'
        pix_local_context = getPixContextSem([uni_edge_pix_x_extended + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                                                   uni_edge_pix_y_extended + LOCAL_CONTEXT_MAP_APPEND_WIDTH, \
                                                   context_map, local_context_paras, None, None])
#     print 'pix_local_context shape',pix_local_context.shape
#     print 'pix_local_context', pix_local_context[:2, :]
    
    return uni_edge_pix_x_extended, uni_edge_pix_y_extended, \
        grad_mag_in_img_L[uni_edge_pix_y_extended, uni_edge_pix_x_extended], \
        grad_mag_enh_img_L[uni_edge_pix_y_extended, uni_edge_pix_x_extended], pix_local_context
        

def compute_img_global_L_transform(in_args):
    in_img_filenm, enh_img_filenm, paras = in_args[0], in_args[1], in_args[2]
    
    
    in_img_dir, enh_img_dir, num_control_points, fredo_image_processing = \
    paras['in_img_dir'], paras['enh_img_dir'], \
    paras['num_control_points'], paras['fredo_image_processing']
    
    in_img_path, enh_img_path = \
    os.path.join(in_img_dir, in_img_filenm), \
    os.path.join(enh_img_dir, enh_img_filenm)
#     print 'compute_img_global_L_transform in_img_path:%s' %\
#     in_img_path
    
    if fredo_image_processing == 1:
        in_img, enh_img = read_tiff_16bit_img_into_LAB(in_img_path, 1.5, False), \
        read_tiff_16bit_img_into_LAB(enh_img_path, 0, False)
    else:
        in_img, enh_img = read_tiff_16bit_img_into_LAB(in_img_path), \
        read_tiff_16bit_img_into_LAB(enh_img_path)
    h, w, c = in_img.shape[0], in_img.shape[1], in_img.shape[2]
    
    in_img_L, enh_img_L = in_img[:, :, 0], enh_img[:, :, 0]
    bspline = get_BSpline_curve(in_img_L.flatten(), enh_img_L.flatten(), \
                                num_control_points, 0.0, 100.0)
    control_points = bspline.get_coeffs()
    print 'control_points shape', control_points.shape
    return control_points
    

def div_up(a, b):
    return (a + b - 1) / b

def min(a, b):
    return a if a <= b else b

def compute_grad_mag_angle(gradx, grady):
    h, w = gradx.shape[0], gradx.shape[1]
    grad_mag = np.sqrt(gradx ** 2 + grady ** 2)

    grad_angle = np.zeros((h, w), dtype=np.single)
    for i in range(h):
        for j in range(w):
            grad_angle[i, j] = math.atan2(grady[i, j], gradx[i, j])
#     gradx2,grady2=grad_mag*np.cos(grad_angle),grad_mag*np.sin(grad_angle)
#     diffx=np.sum(np.abs(gradx-gradx2))
#     diffy=np.sum(np.abs(grady-grady2))
#     print 'diffx,diffy',diffx,diffy
#     print 'gradx',gradx[:2,:5]
#     print 'grady',grady[:2,:5]
#     print 'gradx2',gradx2[:2,:5]
#     print 'grady2',grady2[:2,:5]
    
    return grad_mag, grad_angle

def compute_divergence_from_gradient(grad_mag, grad_angle):
    st_time = time.time()
    assert grad_mag.shape == grad_angle.shape
    h, w, c = grad_mag.shape[0], grad_mag.shape[1], grad_mag.shape[2]
#     print 'compute_divergence_from_gradient grad_mag 1st channel min,max',\
#     np.min(grad_mag[:,:,0]),np.max(grad_mag[:,:,0])
#     print 'compute_divergence_from_gradient grad_angle 1st channel min,max',\
#     np.min(grad_angle[:,:,0]),np.max(grad_angle[:,:,0])
#     print 'h,w,c',h,w,c
    if 0:
        grad_mag, grad_angle = grad_mag.reshape((h * w * c)), grad_angle.reshape((h * w * c))
        div = np.zeros((h * w * c), dtype=np.single)
        assert c == 3
        for i in range(0, h * w):
            x, y = i % w, i / w
            if x >= 1 and y >= 1:
                index = i * c
                for ch in range(c):
                    div[index] = grad_mag[index] * math.cos(grad_angle[index]) - \
                    grad_mag[index - c] * math.cos(grad_angle[index - c]) + \
                    grad_mag[index] * math.sin(grad_angle[index]) - \
                    grad_mag[index - c * w] * math.sin(grad_angle[index - c * w])
                    index += 1
        div = div.reshape((h, w, c))            
    else:
        div = np.zeros((h, w, c), dtype=np.single)
        div[1:h, 1:w, :] = grad_mag[1:h, 1:w, :] * np.cos(grad_angle[1:h, 1:w, :]) - \
        grad_mag[1:h, 0:(w - 1), :] * np.cos(grad_angle[1:h, 0:(w - 1), :]) + \
        grad_mag[1:h, 1:w, :] * np.sin(grad_angle[1:h, 1:w, :]) - \
        grad_mag[0:(h - 1), 1:w, :] * np.sin(grad_angle[0:(h - 1), 1:w, :])
    
        
#     div[0,:,:]=div[1,:,:]
#     div[:,0,:]=div[:,1,:]
#     print 'compute_divergence_from_gradient div 1st channel min,max',\
#     np.min(div[:,:,0]),np.max(div[:,:,0])
    elapsed = time.time() - st_time
#     print 'compute_divergence_from_gradient elapsed:%f' % elapsed
    return div


def find_image_files(src_dir, img_pattern):
    all_img_files = [file for file in os.listdir(src_dir)\
                if os.path.isfile(os.path.join(src_dir, file))\
                and fnmatch.fnmatch(file, img_pattern)]
    all_img_files = sorted(all_img_files)
    return all_img_files   

def findSemIntegralFiles(src_dir):
    context_files = [file for file in os.listdir(src_dir)\
                           if os.path.isfile(os.path.join(src_dir, file))\
                           and fnmatch.fnmatch(file, "*.mat")]
    context_files = sorted(context_files, key=lambda file_name:int(file_name[:-4]))
    return context_files
#             print self.in_context_files    
# copy image files into destination dir based on a text file containing image ids
def choose_and_copy_images(img_id_txt, img_pattern, src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if 0:
        all_img_files = findSemIntegralFiles(src_dir)
        print 'context_files', all_img_files
    else:
        all_img_files = find_image_files(src_dir, img_pattern)

    id_file = open(img_id_txt, 'r')
    for line in id_file:
        img_id = int(line) - 1
        print 'img id:%d img name:%s' % (img_id, all_img_files[img_id])
        src_file_path = os.path.join(src_dir, all_img_files[img_id])
        dest_file_path = os.path.join(dest_dir, all_img_files[img_id])
        if sys.platform == 'win32':
            command = ['copy', src_file_path, dest_file_path]            
        else:
            command = ['cp', src_file_path, dest_file_path]
            
        command = ' '.join(command)
        print command
        call(command, shell=True)
    id_file.close()

def listFileNames(dir,outFilePath,pattern='*.*'):
    files = [file for file in os.listdir(dir)\
             if os.path.isfile(os.path.join(dir,file))\
             and fnmatch.fnmatch(file, pattern)]
    print 'find %d matching files' % len(files)
    outFile=open(outFilePath,'w')
    for file in files:
        outFile.write('%s\n' % file[:-4])
    outFile.close()
        
    
def read_image_id_txt_file(img_id_txt, index_shift=1):
    img_id_file = open(img_id_txt, 'r')
    img_id = []
    for line in img_id_file:
#         print 'line',line
        img_id += [int(line) - index_shift]  # from 1-based index to 0-based index
    img_id_file.close()
    return list(img_id) 

def readImgNameFlie(imgNmFile,allImgNms):
    nameToId={}
    id=0
    for imgNm in allImgNms:
        nameToId[imgNm[:-4]]=id
        id+=1
    
    imgIds=[]
    imgNms = open(imgNmFile, 'r')
    for line in imgNms:
        imgIds += [nameToId[line.rstrip('\r\n')]]
    imgNms.close()
    return imgIds

def write_image_id_txt_file(img_id_txt, ids, index_shift=1):
    id_file = open(img_id_txt, 'w')
    for i in range(len(ids)):
        id_file.write('%d\n' % (ids[i] + index_shift))
    id_file.close()

# given train image ids, find out test image ids
def from_train_idx_get_test_idx(num_imgs, in_train_id_txt, out_test_id_txt):
    all_img_ids = set(range(num_imgs))
    tr_id = read_image_id_txt_file(in_train_id_txt)
    tr_id = set(tr_id)
    ts_id = all_img_ids - tr_id
    write_image_id_txt_file(out_test_id_txt, list(ts_id))
    
#     tr_id_file = open(in_train_id_txt, 'r')
#     ts_id_file = open(out_test_id_txt, 'w')
#     tr_id = set()
#     for line in tr_id_file:
#         print 'line', line
#         tr_id.add(int(line))  # 1-based index in text file
#     print '%d test images' % len(ts_id)
#     
#     for id in ts_id:
#         ts_id_file.write('%d\n' % (id + 1))
#     ts_id_file.close()
#     tr_id_file.close()

# given all image ids and train image ids, find out test image ids
def from_train_idx_get_test_idx_v2(in_all_img_id_txt, in_train_id_txt, out_test_id_txt):
    all_ids = read_image_id_txt_file(in_all_img_id_txt)
    tr_ids = read_image_id_txt_file(in_train_id_txt)
    ts_ids = set(all_ids) - set(tr_ids)
    ts_ids = list(ts_ids)
    print 'train number: test number:', len(tr_ids), len(ts_ids)
    write_image_id_txt_file(out_test_id_txt, ts_ids)

def random_split_tr_ts_id(in_img_id_txt, num_tr, out_tr_img_id_txt, out_ts_img_id_txt):
    img_ids = np.array(read_image_id_txt_file(in_img_id_txt))
    num_ids = len(img_ids)
    print '%d image ids' % num_ids
    assert num_ids > num_tr
    permu = range(num_ids)
    rd.shuffle(permu)
    tr_ids = img_ids[permu[:num_tr]]
    ts_ids = img_ids[permu[num_tr:]]
    write_image_id_txt_file(out_tr_img_id_txt, tr_ids)
    write_image_id_txt_file(out_ts_img_id_txt, ts_ids)
    
def random_split_tr_ts_id_v2(num_img, num_tr, out_tr_img_id_txt, out_ts_img_id_txt):
    img_ids = np.array(range(num_img))
    print '%d image ids' % num_img
    assert num_img > num_tr
    permu = range(num_img)
    rd.shuffle(permu)
    tr_ids = img_ids[permu[:num_tr]]
    ts_ids = img_ids[permu[num_tr:]]
    write_image_id_txt_file(out_tr_img_id_txt, tr_ids)
    write_image_id_txt_file(out_ts_img_id_txt, ts_ids)
    
    
    
def find_global_ftr_context_hist_NN(tr_id_txt, ts_id_txt, \
                                    img_global_ftr_path, cnn_global_ftr_path, \
                                    in_img_context_dir, context_label_num, context_pad_width, \
                                    save_path, NN_k=5):
    tr_img_id = read_image_id_txt_file(tr_id_txt)
    ts_img_id = read_image_id_txt_file(ts_id_txt)
    print 'num_tr_img', len(tr_img_id)
    num_ts_imgs = len(ts_img_id)
    
    in_context_files = [file for file in os.listdir(in_img_context_dir)\
                           if os.path.isfile(os.path.join(in_img_context_dir, file))\
                           and fnmatch.fnmatch(file, "*.mat")]
    in_context_files = sorted(in_context_files, key=lambda file_name:int(file_name[:-4]))
    print 'find %d context files' % len(in_context_files)
    
    # compute histogram over context feature labels
    num_imgs = len(in_context_files)
    context_hist = np.zeros((num_imgs, context_label_num), dtype=np.single)
    for i in range(len(in_context_files)):
        if i % 100 == 0:
            print 'i:%d' % i
        context_map = scipy.io.loadmat(os.path.join(in_img_context_dir, in_context_files[i]))
        context_map = context_map['maps'].reshape((context_label_num))
        h, w = context_map[0].shape[0], context_map[0].shape[1]
        h2, w2 = h - context_pad_width * 2, w - context_pad_width * 2
        l_hist = np.zeros((context_label_num), dtype=np.single)
        for j in range(context_label_num):
            l_hist[j] = context_map[j][context_pad_width - 1, context_pad_width - 1] + context_map[j][h - context_pad_width - 1, w - context_pad_width - 1]\
            - context_map[j][context_pad_width - 1, w - context_pad_width - 1] - context_map[j][h - context_pad_width - 1, context_pad_width - 1]
        assert h2 * w2 == np.sum(l_hist)
        l_hist /= np.sum(l_hist)
        context_hist[i, :] = l_hist
    
    # load cvpr image global feature
    pix_global_ftr = unpickle(img_global_ftr_path)
    pix_global_ftr = pix_global_ftr['img_global_ftr_l2'].transpose()  # shape: n * 207
    print 'pix_global_ftr shape', pix_global_ftr.shape
    
    # load CNN 4096D global feature
    cnn_global_ftr = unpickle(cnn_global_ftr_path)
    cnn_global_ftr = cnn_global_ftr['data']  # shape: n * 4096
    print 'cnn_global_ftr shape', cnn_global_ftr.shape
    
    context_hist_NN = np.zeros((num_ts_imgs, NN_k), dtype=np.int32)
    context_hist_NN_dist = np.zeros((num_ts_imgs, NN_k), dtype=np.single)
    img_global_ftr_NN = np.zeros((num_ts_imgs, NN_k), dtype=np.int32)
    img_global_ftr_NN_dist = np.zeros((num_ts_imgs, NN_k), dtype=np.single)
    cnn_global_ftr_NN = np.zeros((num_ts_imgs, NN_k), dtype=np.int32)
    cnn_global_ftr_NN_dist = np.zeros((num_ts_imgs, NN_k), dtype=np.single)
    
    
    for i in range(len(ts_img_id)):
        l_id = ts_img_id[i]
        print 'ts img id', l_id
        l_context_hist = context_hist[l_id, :]
        l_img_global_ftr = pix_global_ftr[l_id, :]
        l_cnn_global_ftr = cnn_global_ftr[l_id, :]
        
        l_c_hist_dist = np.sqrt(np.sum((context_hist[tr_img_id, :] - l_context_hist[np.newaxis, :]) ** 2, axis=1))
        l_img_global_ftr_dist = np.sqrt(np.sum((pix_global_ftr[tr_img_id, :] - l_img_global_ftr[np.newaxis, :]) ** 2, axis=1))
        l_cnn_global_ftr_dist = np.sqrt(np.sum((cnn_global_ftr[tr_img_id, :] - l_cnn_global_ftr[np.newaxis, :]) ** 2, axis=1))
        
        c_hist_idx = np.argsort(l_c_hist_dist)
        img_global_ftr_idx = np.argsort(l_img_global_ftr_dist)
        cnn_global_ftr_idx = np.argsort(l_cnn_global_ftr_dist)
#         print 'l_c_hist_dist',l_c_hist_dist
#         print 'c_hist_idx',c_hist_idx
#         print 'l_img_global_ftr_dist',l_img_global_ftr_dist
#         print 'img_global_ftr_idx',img_global_ftr_idx
#         print 'l_cnn_global_ftr_dist',l_cnn_global_ftr_dist
#         print 'cnn_global_ftr_idx',cnn_global_ftr_idx
        
        context_hist_NN[i, :] = tr_img_id[c_hist_idx[:NN_k]]
        context_hist_NN_dist[i, :] = l_c_hist_dist[c_hist_idx[:NN_k]]
        img_global_ftr_NN[i, :] = tr_img_id[img_global_ftr_idx[:NN_k]]
        img_global_ftr_NN_dist[i, :] = l_img_global_ftr_dist[img_global_ftr_idx[:NN_k]]
        cnn_global_ftr_NN[i, :] = tr_img_id[cnn_global_ftr_idx[:NN_k]]
        cnn_global_ftr_NN_dist[i, :] = l_cnn_global_ftr_dist[cnn_global_ftr_idx[:NN_k]]
    
    save_dict = {}
    save_dict['tr_img_id'] = tr_img_id
    save_dict['ts_img_id'] = ts_img_id
    save_dict['context_hist_NN'] = context_hist_NN
    save_dict['context_hist_NN_dist'] = context_hist_NN_dist
    save_dict['img_global_ftr_NN'] = img_global_ftr_NN
    save_dict['img_global_ftr_NN_dist'] = img_global_ftr_NN_dist
    save_dict['cnn_global_ftr_NN'] = cnn_global_ftr_NN
    save_dict['cnn_global_ftr_NN_dist'] = cnn_global_ftr_NN_dist
    pickle(save_path, save_dict)
        
# cluster training images based on global feature. Select those close to cluster centers
# use selected training image to produce train batch data
def select_train_image(tr_id_txt, img_global_ftr_path, cnn_global_ftr_path, \
                       cluster_k=100, iter=200, thres_percentile=0.1, thres_multiply=2.0):
    tr_img_id = read_image_id_txt_file(tr_id_txt)
    # load cvpr image global feature
    pix_global_ftr = unpickle(img_global_ftr_path)
    pix_global_ftr = pix_global_ftr['img_global_ftr_l2'].transpose()  # shape: n * 207
    img_global_ftr_tr = pix_global_ftr[tr_img_id, :]
    print 'pix_global_ftr shape', pix_global_ftr.shape
    
    
    # load CNN 4096D global feature
    cnn_global_ftr = unpickle(cnn_global_ftr_path)
    cnn_global_ftr = cnn_global_ftr['data']  # shape: n * 4096
    cnn_global_ftr_tr = cnn_global_ftr[tr_img_id, :]
    print 'cnn_global_ftr shape', cnn_global_ftr.shape
    
    # zero-mean and make variance unit on each feature dimension
    global_ftr_tr = np.hstack((cnn_global_ftr_tr, img_global_ftr_tr))
    global_ftr_tr_mean = np.mean(global_ftr_tr, axis=1)
    global_ftr_tr = global_ftr_tr - global_ftr_tr_mean[np.newaxis, :]
    global_ftr_tr_std_dev = np.std(global_ftr_tr, axis=1)
    print 'global_ftr_tr_mean', global_ftr_tr_mean
    print 'global_ftr_tr_std_dev', global_ftr_tr_std_dev
    global_ftr_tr = global_ftr_tr / global_ftr_tr_std_dev[np.newaxis, :]
#     global_ftr_tr = scipy.cluster.vq.whiten(global_ftr_tr)
    
    centroids, distortion = scipy.cluster.vq.kmean(global_ftr_tr, cluster_k, iter)
    print 'cluster_k,iter,distortion', cluster_k, iter, distortion
    code, dist = scipy.cluster.vq.vq(global_ftr_tr, centroids)
    print 'code min,max', np.min(code), np.max(code)
    selected_tr_img_id = [None] * cluster_k
    for i in range(cluster_k):
        id = np.nonzero(code == i)
        l_dist = dist[id[0]]
        l_size = len(id)
        print 'cluster %d size:%d' % (i, l_size)
        l_dist_sorted = np.sort(l_dist)
        thres_id = np.round(l_size * thres_percentile + 0.5)
        assert thres_id < l_size
        l_dist_thres = l_dist_sorted[thres_id] * thres_multiply
        id2 = np.nonzero(l_dist < l_dist_thres)
        selected_tr_img_id[i] = tr_img_id[id[0]][id2[0]]
        print 'selected %d out of %d' % (len(id2[0]), len(id[0]))
    ret_tr_img_id = np.hstack(selected_tr_img_id)
    print 'ret_tr_img_id length', len(ret_tr_img_id)
    
    return ret_tr_img_id, global_ftr_tr_mean, global_ftr_tr_std_dev, centroids

# read 'newLab.txt' produced by Poisson reconstruction
def read_Lab_txt_into_Lab_img(img_h, img_w, txt_path):
    img_Lab = np.zeros((img_h, img_w, 3), dtype=np.single)
    f = open(txt_path, 'r')
    for i in range(img_h):
        for j in range(img_w):
            line_str = f.readline()
            words = line_str.split()
            vals = [float(word) for word in words]
            img_Lab[i, j, :] = vals
    f.close()
#         print 'img_Lab',img_Lab[0,0,:]
    # rescale
    scale = np.array([100.0, 128.0, 128.0])
    img_Lab = img_Lab * scale[np.newaxis, np.newaxis, :]
    return img_Lab

# read 'us.txt', the input of Poisson reconstruction
def read_Lab_txt_into_Lab_img_v2(txt_path):
    f = open(txt_path, 'r')
    pixnum = int(f.readline())
    img_w = int(f.readline())
    assert pixnum % img_w == 0
    img_h = pixnum / img_w
    print 'img_h,img_w', img_h, img_w
    img_Lab = np.zeros((img_h, img_w, 3), dtype=np.single)
    
    for ch in range(3):
        for i in range(img_h):
            for j in range(img_w):
#                 line_str = f.readline()
#                 words = line_str.split()
#                 vals = [float(word) for word in words]
                img_Lab[i, j, ch] = float(f.readline())
    f.close()
#         print 'img_Lab',img_Lab[0,0,:]
    # rescale
    scale = np.array([100.0, 128.0, 128.0])
    img_Lab = img_Lab * scale[np.newaxis, np.newaxis, :]
    return img_Lab

        
def get_difference_img(gp1_dir, gp2_dir, different_map_dir, img_pattern='*.png'):
    if not os.path.exists(gp1_dir):
        print 'group 1 dir %s is not found' % gp1_dir
    if not os.path.exists(gp2_dir):
        print 'group 2 dir %s is not found' % gp2_dir
    if not os.path.exists(different_map_dir):
        os.mkdir(different_map_dir)
    
    gp1_imgs = [file for file in os.listdir(gp1_dir)\
                if os.path.isfile(os.path.join(gp1_dir, file)) and\
                fnmatch.fnmatch(file, img_pattern)]
    gp2_imgs = [file for file in os.listdir(gp2_dir)\
                if os.path.isfile(os.path.join(gp2_dir, file)) and\
                fnmatch.fnmatch(file, img_pattern)]
    for gp1_img in gp1_imgs:
        print 'compare image %s' % gp1_img
        if not gp1_img in gp2_imgs:
            print 'image %s is not found in group 2 dir. Exit! ' % gp1_img
            sys.exit()
        if img_pattern == '*.png':
            img1_srgb = scipy.misc.imread(os.path.join(gp1_dir, gp1_img))
            img2_srgb = scipy.misc.imread(os.path.join(gp2_dir, gp1_img))
            h, w, c = img1_srgb.shape[0], img1_srgb.shape[1], img1_srgb.shape[2]
            diff_img = np.uint8(np.abs(np.int32(img1_srgb) - np.int32(img2_srgb)))
    #         print 'diff_map dtype min,max,median', diff_img.dtype,\
            np.min(diff_img), np.max(diff_img), np.median(diff_img)
    #         idx = np.nonzero(diff_img[1:(h - 1), 1:(w - 1), :])
            mean_diff_L2 = np.sum(np.sqrt(np.sum(diff_img ** 2, axis=2))) / np.single(h * w)
            print 'RGB mean_diff_L2:%f' % (mean_diff_L2)
            scipy.misc.imsave(os.path.join(different_map_dir, gp1_img), diff_img)
        elif img_pattern == '*.tif':
            img1Tif=tiff.imread(os.path.join(gp1_dir,gp1_img))
            img2Tif=tiff.imread(os.path.join(gp2_dir,gp1_img))
            diffImg=np.uint16(np.abs(np.float64(img1Tif)-np.float64(img2Tif)))
            tiff.imsave(os.path.join(different_map_dir, gp1_img),diffImg)
        else:
            print '%s image format is not supported'
            return        


def do_srgb_gamma_correcton(in_img_dir, out_img_dir, img_pattern='*.png'):
    in_imgs = [file for file in os.listdir(in_img_dir)\
                if os.path.isfile(os.path.join(in_img_dir, file)) and\
                fnmatch.fnmatch(file, img_pattern)]   
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    
    for in_img in in_imgs:
        print 'do gamma corretion to image %s' % in_img
        in_img_pix = scipy.misc.imread(os.path.join(in_img_dir, in_img))
        out_img_pix = gamma_correction_SRGB(in_img_pix)
        scipy.misc.imsave(os.path.join(out_img_dir, in_img), out_img_pix)

# clamp r,g,b values to range [0,1]
def clamp_sRGB_img(img_srgb):    
    idx = np.nonzero(img_srgb < 0)
    img_srgb[idx[0], idx[1], idx[2]] = 0
    idx = np.nonzero(img_srgb > 1)
    img_srgb[idx[0], idx[1], idx[2]] = 1
    return img_srgb
    
def read_fredo_errors(err_txt):
    err_file = open(err_txt, 'r')
    errs = []
    for line in err_file:
#         print 'line',line
        errs += [float(line)] 
    err_file.close()
    return list(errs)
    
    
if __name__ == "__main__":
    if 1:
        listFileNames(r'D:\yzc\proj\cnn-image-enhance\data\xproIII\xproIIIInImgAutotunedTif',\
                      r'D:\yzc\proj\cnn-image-enhance\data\xproIII\inimgs.txt',\
                      '*.tif')
        sys.exit()
    if 0:
        nTs, nTr = 22, 43
        trIds, tsIds = range(nTs, nTs + nTr), range(0, nTs)
        write_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\xproIII\xproIII_train_id_2.txt', trIds)
        write_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\xproIII\xproIII_test_id_2.txt', tsIds)        
        sys.exit()
    if 0:
        # MSR second effect. 50 images, 10 folds, 10 train/test splits
        ids = range(50)
        rd.shuffle(ids)
        for i in range(10):
            ts_ids = ids[i * 5:i * 5 + 5]
            tr_ids = list(set(ids) - set(ts_ids))
            write_image_id_txt_file\
            (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\secondEffect\secondeffect_train_id_%d.txt' % (10 + i), tr_ids)
            write_image_id_txt_file\
            (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\secondEffect\secondeffect_test_id_%d.txt' % (10 + i), ts_ids)
        sys.exit()
                        
            
    
    if 0:
        # for second effect, generate test id file
        img_tif_files = find_image_files(r'/home/zyan3/proj/cnn-image-enhance/data/MSR_effect/secondEffect/secondEffect_input_tiff_50', \
                         '*.tif')
        print 'img_tif_files', img_tif_files
        
        test_id_2 = []
        target_test_id = read_image_id_txt_file(r'/home/zyan3/proj/cnn-image-enhance/data/MSR_effect/secondEffect/secondeffect_test_id_2_target.txt')
        print 'target_test_id', target_test_id
        for t_id in target_test_id:
            t_id = t_id + 1
            for j in range(len(img_tif_files)):
                tid_file_id = int(img_tif_files[j][:5])
                if t_id == tid_file_id:
                    test_id_2 += [j]
                    break
        print 'test_id_2', test_id_2
        write_image_id_txt_file(r'/home/zyan3/proj/cnn-image-enhance/data/MSR_effect/secondEffect/secondeffect_test_id_2.txt', \
                                test_id_2)
        from_train_idx_get_test_idx(50, r'/home/zyan3/proj/cnn-image-enhance/data/MSR_effect/secondEffect/secondeffect_test_id_2.txt', \
                                    r'/home/zyan3/proj/cnn-image-enhance/data/MSR_effect/secondEffect/secondeffect_train_id_2.txt')
        sys.exit()
    
    if 0:
        top40_id = read_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_img_id_top40.txt')
        
        os.chdir(r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\context\Parts_manual')
        for id in top40_id:
            mat_fn = '%d' % (id + 1) + '.mat'
            new_mat_fn = '%d' % (id + 5000 + 1) + '.mat'
            print mat_fn, new_mat_fn
            command = ['copy', mat_fn, new_mat_fn]
            command = ' '.join(command)
            print command
            call(command, shell=True)


        
        sys.exit()
    
    if 0:
        ids = range(70, 110)
        write_image_id_txt_file(r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_40_regularize_img_id_small.txt', \
                                ids)
        sys.exit()
    
    if 0:
        # change the file name. increase the image id by 5000
        img_dir = r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\context\precomputed_context_ftr_40_regularize'
        os.chdir(img_dir)
        img_files = [file for file in os.listdir(img_dir)]
        print 'num:%d' % len(img_files)
        for img_f in img_files:
#             num = int(img_f[:-4])
            num = int(img_f[1:5])
            print 'num', num
            num += 5000
#             img_f_new = ('%04d' % num) + '.mat'
            img_f_new = img_f[0] + ('%04d' % num) + img_f[5:]
            print 'img_f %s,img_f_new:%s' % (img_f, img_f_new)
            command = ['move', img_f, img_f_new]
            command = ' '.join(command)
            print command
            call(command, shell=True)
        sys.exit()
        
    if 0:
        id_50 = read_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_img_id.txt', \
         0)
        id_50 = np.array(id_50)
        id_20 = read_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_20_new_id.txt', \
         0)
        id_20 = np.array(id_20)
        all_id = np.concatenate((id_50, id_20))
        print 'all_id', all_id
        sorted_id = np.sort(all_id)
        print 'sorted_id', sorted_id
        num_id = 70
        new_ids = np.zeros((num_id), dtype=np.int32)
        for i in range(num_id):
            l_id = all_id[i]
            l_idx = np.nonzero(sorted_id == l_id)
            new_ids[i] = l_idx[0][0]
        print 'new_ids', new_ids
        
        write_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_img_id_small.txt', \
         new_ids[:50])
        write_image_id_txt_file\
        (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_20_new_id_small.txt', \
         new_ids[50:70])    
        sys.exit()
    
    if 0:
        fredo_errors = read_fredo_errors(r'/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/fredo_errors.txt')
        fredo_errors = np.array(fredo_errors)
        print 'fredo_errors length', fredo_errors.shape[0]
        print 'fredo error mean', np.mean(fredo_errors)
        print 'fredo error max', np.max(fredo_errors)
    
        enh_summ = unpickle\
        (r'/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/export_tiff/convnet_checkpoints/ConvNet__2014-01-10_11.35.05zyan3_summary/InputWithExpertCWhiteBalanceMinus1.5_enhanced_expert_c_pixel/enhance_summary')
        print 'L_dist length', enh_summ['L_dist'].shape[0]
        print 'L dist mean', np.mean(enh_summ['L_dist'])
        print 'L dist max', np.max(enh_summ['L_dist'])
        
        dict1 = {}
        dict1['fredo_errors_L'] = fredo_errors
        dict1['out_error_L'] = enh_summ['L_dist']
        scipy.io.savemat(r'/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/errors_comparison.mat', dict1)
        
        sys.exit()
        
    if 0:
        img_Lab = read_Lab_txt_into_Lab_img\
        (332, 500, r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_tiff\convnet_checkpoints\ConvNet__2014-01-08_04.09.02yzyu-server2_summary\poisson_data\a4671-DSC_0026\us.txt')
        img_srgb = color.lab2rgb(img_Lab)
        img_srgb = clamp_sRGB_img(img_srgb)
        
        scipy.misc.imsave(r'C:\Users\yzc\Pictures\a4671_zyan3.png', img_srgb)
        print 'r min max,g min,max b min,max', np.min(img_srgb[:, :, 0]), \
        np.max(img_srgb[:, :, 0]), np.min(img_srgb[:, :, 1]), np.max(img_srgb[:, :, 1]), \
        np.min(img_srgb[:, :, 2]), np.max(img_srgb[:, :, 2])
        mpplot.figure()
        mpplot.imshow(img_srgb)
        mpplot.show()
        sys.exit()

    if 0:
        choose_and_copy_images\
        (r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\firsteffect_img_id_top40.txt', \
         '*.mat', r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\context\precomputed_context_ftr', \
        r'D:\yzc\proj\cnn-image-enhance\data\MSR_effect\firstEffect\context\precomputed_context_ftr_40_regularize')
        sys.exit()
    
    if 0:
        random_split_tr_ts_id\
        (r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_img_id.txt', \
         40, r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_train_id_2.txt', \
         r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_test_id_2.txt')
        sys.exit()
    
    if 0:
        in_img_Lab = read_Lab_txt_into_Lab_img_v2\
        (r'D:\yzc\Dropbox\Public\PoissonEditing_exe\us.txt')
        out_img_Lab = read_Lab_txt_into_Lab_img\
        (333, 500, r'D:\yzc\Dropbox\Public\PoissonEditing_exe\newLab.txt')
        in_img_srgb = color.lab2rgb(in_img_Lab)
        in_img_srgb = clamp_sRGB_img(in_img_srgb)
        out_img_srgb = color.lab2rgb(out_img_Lab)
        out_img_srgb = clamp_sRGB_img(out_img_srgb)
        scipy.misc.imsave(r'D:\yzc\Dropbox\Public\PoissonEditing_exe\us.png', in_img_srgb)
        scipy.misc.imsave(r'D:\yzc\Dropbox\Public\PoissonEditing_exe\newLab.png', out_img_srgb)
        diff_img_srgb = np.abs(in_img_srgb - out_img_srgb)
        scipy.misc.imsave(r'D:\yzc\Dropbox\Public\PoissonEditing_exe\diff.png', diff_img_srgb)
        sys.exit()
    
    if 1:
        print 'get_difference_img'
        cp_dir = r'D:\yzc\proj\cnn-image-enhance\data\xproIII\convnet_checkpoints'
        cp1=r'ConvNet__2014-05-14_05.41.10yzyu-server2'
        cp2=r'ConvNet__2014-05-14_05.41.43yzyu-server2'
        get_difference_img\
        (os.path.join\
         (cp_dir, cp1+r'_summary\superpixel'), \
         os.path.join
         (cp_dir, cp2+r'_summary\superpixel'), \
         os.path.join\
         (cp_dir, cp2+r'_summary\superpixeldiff_'+cp1),\
         img_pattern='*.tif')
        sys.exit()
    
    if 0:
        find_global_ftr_context_hist_NN(r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_tiff\train_idx.txt', \
                                        r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_tiff\test_idx.txt', \
                                        r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_tiff\InputAsShotZeroed\pix_global_ftr', \
                                        r'D:\yzc\proj\cuda-convnet-plus\cuda-convnet-data\imagenet\12_challenge\convnet_checkpoint\ConvNet__2013-10-26_07.58.01yzyu-server2_summary\fc4096_2\39.199\data_batch_601', \
                                        r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\context\Parts', \
                                        20, 100, \
                                        r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_tiff\global_ftr_context_hist_NN', NN_k=5)
        sys.exit()
    if 0:
        test_id = read_image_id_txt_file(r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\eccv_random250_testindex_raw.txt', 0)
        f = open(r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\eccv_random250_test_id.txt', 'w')
        for id in test_id:
            f.write('%d\n' % (id + 1))
        f.close()
        sys.exit()
    
    if 1:
        from_train_idx_get_test_idx\
        (5000, r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_train_id_3.txt', \
         r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_train_id_3_rest.txt')
        sys.exit()
    
    if 1:
        from_train_idx_get_test_idx_v2\
        (r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_img_id.txt', \
         r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_test_id_3.txt', \
         r'D:\yzc\proj\cnn-image-enhance\data\firsteffect_train_id_3.txt')
        sys.exit()
    
    
    if sys.platform == 'win32':
        export_path = 'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_2'
    else:
        export_path = '/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/export_2'
    
        
    in_img_dir = os.path.join(export_path, 'InputAsShotZeroed')
    enh_img_dir = os.path.join(export_path, 'expert_c')
    ori_img_fn = 'a0063-IMG_4185.jpg'
    eng_img_fn = 'a0063-IMG_4185.jpg'
    saliency_fn = os.path.join(export_path, \
                               'InputAsShotZeroed_HC/a0063-IMG_4185_HC.png')
    
    
    in_img = read_img(os.path.join(in_img_dir, ori_img_fn))
    enh_img = read_img(os.path.join(enh_img_dir, eng_img_fn))
    h, w = in_img.shape[0], in_img.shape[1]
#     if sys.platform == 'linux2':
#         in_img = in_img[::-1, :, :]
#         enh_img = enh_img[::-1, :, :]
    
#     mpplot.figure(0)
#     mpplot.imshow(in_img)
# 
#     mpplot.figure(1)
#     mpplot.imshow(enh_img)    
#         
#     sc_map = scipy.ndimage.imread(saliency_fn)
#     mpplot.figure(2)
#     sc_img = mpplot.imshow(sc_map, cmap=mpplot.cm.get_cmap('Greys_r'))
    
    
    paras = {}
    paras['in_img_dir'], paras['enh_img_dir'], \
    paras['patch_half_size'], paras['nbSize'], paras['stride'], paras['meanResidueThres'], \
    paras['gauSigma'], paras['lsLambda'], paras['lsAtol'], paras['saliency_power'], \
    paras['use_saliency'] = \
    in_img_dir, enh_img_dir, 64, 2, 6, 2, 1.2, 1e0, 1e-6, 0.7, 0
        
    
    args = [ori_img_fn, eng_img_fn, saliency_fn, paras]
    numPos, posx, posy, mappings = processImgs(args)
    print '%d valid patches out of %d' % (len(posx), numPos)
    print 'mappings (max,min) mag:%f,%f' % (np.max(np.abs(mappings)), \
                                            np.min(np.abs(mappings)))
 
#     # visualize sampling positions
    in_img = scipy.misc.imread(os.path.join(in_img_dir, ori_img_fn))
    for p in range(len(posx)):
        px, py = posx[p], posy[p]
        in_img[py - 3:py + 4, px - 3:px + 4, 1:3] = 0
    mpplot.figure(4)
    mpplot.imshow(in_img)
    
    
    # plot histogram of mapping coefficients magnitudes
    mappingAbsSum = np.zeros((len(posx)))
    for i in range(len(posx)):
        mappingAbsSum[i] = np.sum(np.abs(mappings[i, :, :]))
    mpplot.figure(5)
    mpplot.hist(mappingAbsSum, 300)
        
#     visualize mappings
#     posx_2,posy_2=get_samples_pos(h,w,paras['patch_half_size'],1,None)
#     h2,w2=posx_2.shape[0],posx_2.shape[1]
#     posx_2,posy_2=posx_2.flatten(),posy_2.flatten()
#     estimator = ColorMappingEstimator(paras['nbSize'], paras['meanResidueThres'], \
#                                       paras['gauSigma'], paras['lsLambda'], paras['lsAtol'])
#     mappings, goodPos = estimator.get_color_mapping(in_img, enh_img, posx_2, posy_2)
#     print '%d out of %d positions are good' % (len(goodPos),len(posx_2))
#     mappings=mappings.reshape((h2*w2,mappings.shape[1]*mappings.shape[2]))
#     min,max=np.min(mappings,axis=0),np.max(mappings,axis=0)
#     mappings = (mappings-min) / (max-min)
#     mappings = mappings.reshape((h2,w2,mappings.shape[1]))
#     print 'mappings shape',mappings.shape
#     
#     figid=6
#     for i in range(10):
#         mpplot.figure(figid)
#         mpplot.imshow(mappings[:,:,3*i:3*i+3])
#         figid+=1
        
    print mappings[0, :, :]
    
    mpplot.show()    
    
    
    
