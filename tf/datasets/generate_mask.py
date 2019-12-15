import os 
import numpy as np
import cv2
import math

import matplotlib
#matplotlib.get_backend()
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

imagesets=['trainval','test']
dataset_version='v2'

for imageset in imagesets:
    labels_path = 'D:/data/gaofen/gaofen_merge/{}/{}/labels'.format(dataset_version,imageset)
    mask_save_path = 'D:/data/gaofen/gaofen_merge/{}/{}/masks'.format(dataset_version,imageset)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    for idx, label_file in enumerate(os.listdir(labels_path)):
        print(idx,label_file)
        filename=label_file.split('.txt')[0]
        label_file=os.path.join(labels_path,filename+'.txt')
        mask_save_file=os.path.join(mask_save_path,filename+'_mask.png')
        
        im=np.zeros((1024,1024),dtype="uint8")
        lines = open(label_file, 'r').readlines()
        for line in lines:
            thetaobb = [float(xy) for xy in line.rstrip().split(' ')[:-1]]
            x_cor=[]
            y_cor=[]
            xy_cor=[]
            [cx,cy,w,h,theta]=thetaobb

            x1=cx-w/2
            y1=cy-h/2

            x0=cx+w/2
            y0=cy-h/2

            x3=cx+w/2
            y3=cy+h/2

            x2=cx-w/2
            y2=cy+h/2

            cos_theta=math.cos(theta)
            sin_theta=math.sin(theta)

            x0n=(x0-cx)*cos_theta-(y0-cy)*sin_theta+cx
            y0n=(x0-cx)*sin_theta+(y0-cy)*cos_theta+cy

            x1n=(x1-cx)*cos_theta-(y1-cy)*sin_theta+cx
            y1n=(x1-cx)*sin_theta+(y1-cy)*cos_theta+cy

            x2n=(x2-cx)*cos_theta-(y2-cy)*sin_theta+cx
            y2n=(x2-cx)*sin_theta+(y2-cy)*cos_theta+cy

            x3n=(x3-cx)*cos_theta-(y3-cy)*sin_theta+cx
            y3n=(x3-cx)*sin_theta+(y3-cy)*cos_theta+cy

            xy_cor.append((x0n,y0n))
            xy_cor.append((x1n,y1n))
            xy_cor.append((x2n,y2n))
            xy_cor.append((x3n,y3n))

            cv2.polylines(im,np.int32([xy_cor]),1,255)
            cv2.fillPoly(im,np.int32([xy_cor]),255)

        mask=im
        cv2.imwrite(mask_save_file,mask)
        #plt.imshow(mask)
        #plt.show()