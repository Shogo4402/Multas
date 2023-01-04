import math
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import sys
sys.path.append("/content/Multas")
import tool.preparation_dataset as pred

def rmse_self(y_test,prediction):
  num = y_test.shape[0]
  err = y_test-prediction
  rmse = math.sqrt(np.sum(np.square(err))/num)
  return rmse

def mae_self(y_test,prediction):
  num = y_test.shape[0]
  mae = np.sum(np.abs(y_test-prediction))/num
  return mae

def yyplot(y_obs, y_pred):
  yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
  ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
  fig = plt.figure(figsize=(4, 4))
  plt.scatter(y_obs, y_pred)
  plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
  plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
  plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
  plt.xlabel('y_observed', fontsize=24)
  plt.ylabel('y_predicted', fontsize=24)
  plt.title('Observed-Predicted Plot', fontsize=24)
  plt.tick_params(labelsize=16)
  plt.show()
  return fig
  
def correct_rate(prediction,test):
  count = 0
  for i in range(test.shape[0]):
    if prediction[i].argmax()==test[i].argmax():
      count += 1
  return count/test.shape[0]

def result_super_softmax(y_arr,no,MIN_MAX,PN):
  x2 = np.linspace(MIN_MAX[no][0], MIN_MAX[no][1], PN[no])
  seki = y_arr*x2
  return np.sum(seki)

def get_poslist(pos_datas):
  li = [[],[],[]]
  for pos in pos_datas:
    for j in range(3):
      pos_tmp = float(pos.split("\t")[3+j])
      li[j].append(pos_tmp)
  return li

def true_result(PRE,MIN_MAX,PN):
  fr = open("./dataset_PosEst/valid/"+"position_valid.txt", 'r')
  pos_datas = fr.readlines()
  fr.close()
  image_files = natsorted(glob.glob("/content/dataset_PosEst/valid/"+"image/*"))
  pos_datas = pred.datalist_sorted(image_files,pos_datas)
  pos_datas = get_poslist(pos_datas)
  Err_list = []
  list_pre = []
  list_true = []
  c = 0
  for pre,pos in zip(PRE,pos_datas):
    err_li = []
    pre_li = []
    true_li = []
    for pre_vp,vt in zip(pre,pos):
      vp = result_super_softmax(pre_vp,c,MIN_MAX,PN)
      err = abs(vp-vt)
      err_li.append(err)
      pre_li.append(vp)
      true_li.append(vt)
    Err_list.append(err_li)
    list_pre.append(pre_li)
    list_true.append(true_li)
    c+=1
  return np.array(Err_list),np.array(list_pre),np.array(list_true)
