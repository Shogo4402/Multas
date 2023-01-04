import numpy as np
import cv2
import math
import glob
from natsort import natsorted
from scipy.stats import norm

def img_change(img,iw,ih):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # ガウシアンぼかし
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
  img = np.where(img < 0, 0, img)
  img = cv2.resize(img, dsize=(iw, ih))
  #min_v = np.min(img)
  max_v = np.max(img)
  img = img/max_v
  #img = cv2.resize(img, dsize=(iw, ih))
  return img
  
def datalist_sorted(input_images,datalist):
  new_datalist = []
  for iiname in input_images:
    d = iiname.split("_")[3:6]
    no = d[2].replace(".png","") 
    for data in datalist:
      d2 = data.split("\t")
      no2 = d2[6].replace("\n","")
      d2 = d2[1:3]
      if d[0]==d2[0] and d[1]==d2[1] and no == no2:
        new_datalist.append(data)
  return new_datalist

def super_gaussian(pos,no,MIN_MAX,PN):
    scale = 0.1
    x = np.linspace(MIN_MAX[no][0], MIN_MAX[no][1], PN[no])
    y = norm.pdf(x, loc=pos, scale=scale)
    max_y = np.sum(y)
    hoge = y/max_y
    hoge = hoge.reshape(PN[no], -1)
    y_arr = np.sum(hoge,axis=1)
    return y_arr

def pos_gaussian_list(poslist,no,One_or_Pro,MIN_MAX,PN):
    tea_data = []
    if One_or_Pro==0:
      for sx in poslist:
        sg = super_gaussian(sx,no,MIN_MAX,PN)
        max_index = np.argmax(sg)
        sg = np.zeros(PN[no])
        sg[max_index] = 1
        tea_data.append(sg)
    elif One_or_Pro == 1:
      for sx in poslist:
        sg = super_gaussian(sx,no,MIN_MAX,PN)
        tea_data.append(sg)
    else:
      print(0/0)
    return np.array(tea_data)

def making_dataset(folder_name,One_or_Pro=None,MIN_MAX=None,PN=None):
    #image files download
    kind = folder_name.split("/")[3]
    image_files = natsorted(glob.glob(folder_name+"image/*"))
    fr = open(folder_name+"position_"+kind+".txt", 'r')
    pos_datas = fr.readlines()
    fr.close()
    pos_datas = datalist_sorted(image_files,pos_datas)
    posx_list = []
    posy_list = []
    post_list = []
    for pd in pos_datas:
      posx_list.append(float(pd.split("\t")[3]))
      posy_list.append(float(pd.split("\t")[4]))
      post_list.append(float(pd.split("\t")[5]))
      
    ytrain_list = []
    ytrain_list.append(pos_gaussian_list(posx_list,0,One_or_Pro,MIN_MAX,PN))
    ytrain_list.append(pos_gaussian_list(posy_list,1,One_or_Pro,MIN_MAX,PN))
    ytrain_list.append(pos_gaussian_list(post_list,2,One_or_Pro,MIN_MAX,PN))

    input_list = []
    for imagef in image_files:
        img = cv2.imread(imagef)
        img =img_change(img)
        input_list.append(img)

    #dataset 作成
    input1 = np.array(input_list)
    return input1,ytrain_list
