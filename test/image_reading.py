# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:35:35 2020

@author: Jack
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image,ImageDraw 
import re
result_path = r'D:\workplace\量化联赛第二期参考资料\Results_V1_2\TrainingType_1'

marked = pd.read_csv(r'D:\workplace\量化联赛第二期参考资料\marked'+'\Type_1.csv',index_col=['IMG_ID'] )

def stack_img(img_path, abs_path=result_path):
    img = Image.open( abs_path + '\\'+ img_path )
    img = img.convert('1')
    img = np.array(img)+0
    img_ravel = img.ravel()
    return img_ravel  #返回 一维  1*1024

filenames = os.listdir(result_path)
data = np.full( (len(filenames), 32*32),np.nan)
filenames_split =[]
y = []
for i in (marked.index).tolist():
    ind,null = i.split('.jpg')
    num_str = str( marked.loc[i].MARK)
    for j in filenames:
        try:
            if( j.startswith(ind) ):
                y.append( int( num_str[ int( j.split('.png')[0][-1]) ] ) )
                data[( len(y)-1),:] = stack_img( j, )
        except:
            pass


image = Image.open(result_path + '\\Type_1\\0bae515e-4e7b-11ea-9e7c-001a7dda7113.jpg')   #读取图片文件
imgpath= result_path + '\\Type_1\\0bae515e-4e7b-11ea-9e7c-001a7dda7113.jpg'#图片路径
#读取图片RGB信息到array列表
im = Image.open(imgpath)#打开图片到im对象
im_arr = np.array(im)
w,h=im.size #读取图片宽、高# 
img = im.convert('RGB')#将im对象转换为RBG对象
im.show()
img = np.array(img) #将图片以数组的形式读入变量
img0 = img[:,:,0].ravel() #temp0 = img0[ y==0]
img1 = img[:,:,1].ravel()
img2 = img[:,:,2].ravel()
plt.hist(img0,50)
plt.hist(img1,50)
plt.hist(img2,50)
data = pd.DataFrame( {0:img0 ,1:img1,2:img2,} )  #1:img1 ,2:img2
x = data.as_matrix()
