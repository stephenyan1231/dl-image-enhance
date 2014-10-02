import sys
sys.path.append('/home/zyan3/proj/cuda-convnet-plus/cuda-convnet-plus/py')

import os
import fnmatch
import scipy.misc
import numpy as np
from subprocess import call

from util import *

if __name__ == '__main__':
    ''' use external tool to segment all images '''
    if sys.platform =='win32':
        in_img_dir=r'D:\yzc\proj\cnn-image-enhance\data\mit_fivek\export_tiff\expert_d_ppm'
        executable =r'D:\yzc\proj\cnn-image-enhance\tools\segment\segment '
    else:
        in_img_dir = r'/home/zyan3/proj/cnn-image-enhance/data/mit_fivek/export_2/InputAsShotZeroed_ppm'
        executable = r"/home/zyan3/proj/cnn-image-enhance/tools/segment/segment "
#     if not os.path.exists(out_seg_dir):
#         os.mkdir(out_seg_dir)
            
    in_imgs_names = [file for file in os.listdir(in_img_dir) if os.path.isfile(os.path.join(in_img_dir, file)) and fnmatch.fnmatch(file, '*.ppm')]
    in_imgs_names = sorted(in_imgs_names)
    print 'find %d input images' % len(in_imgs_names)
    sigma,k,min,max=0.25,3,10,20
    
#     id=195
#     print '%d th image %s' % (id+1,in_imgs_names[id])
    
    for i in range(len(in_imgs_names)):
#     for i in range(2):
        print '%d out of %d images' % (i+1,len(in_imgs_names))
        in_img_name=in_imgs_names[i]
        vis_file_name=in_img_name[:-4]+".vis"
        seg_file_name=in_img_name[:-4]+".seg"
                
        command = [executable,str(sigma),str(k),str(min),\
                   str(max),"\"" + os.path.join(in_img_dir,in_img_name)+"\"",\
                   "\""+os.path.join(in_img_dir,vis_file_name)+"\""]
         
        print 'command',command
        command = " ".join(command)
        call(command, shell=True)
    meta={}
    meta['sigma']=sigma
    meta['k']=k
    meta['min']=min
    meta['max']=max
    pickle(os.path.join(in_img_dir,'seg_meta'),meta)
      