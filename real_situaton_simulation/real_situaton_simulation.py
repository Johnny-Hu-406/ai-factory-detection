import cv2
import glob
# import matplotlib.pyplot as plt
import math
import numpy as np
import os

from data_aug.data_aug import *
from data_aug.bbox_util import *

'''
原圖>切割出金手指>切成數等份(保留切割時的位置資訊)
自動產生資料增強>進行訓練
得到inference結果>回推bbox
'''
H=608
W=608
CONV_OUT_PATH = "/home/nvidia/Desktop/github/ai-factory-detection/real_situaton_simulation/cut_Test_img"

blcbackgroung = np.zeros((H,W,3), np.uint8)
img_path = "/home/nvidia/Desktop/github/ai-factory-detection/real_situaton_simulation/real_img"

pcb_loc_dc = {}
fin_loc_dc = {}
cutimg_loc_dc = {}


def y_brightness(thimg ,brightness=100, diff=False):
    '''
    diff = True, 亮度偵測會在接觸到第一個符合閥值的px後,繼續往下尋找
    diff = False, 亮度偵測會在接觸到第一個符合閥值的px後,從另外一個邊界尋找

    舉例: y = 0~30, 
    diff= True,30....>28>27>26>......
    diff= False,30....>28>0>1>2>....
    '''
    touch_botten = False
    thimg = cv2.cvtColor(thimg, cv2.COLOR_GRAY2RGB)
    thimg = cv2.cvtColor(thimg, cv2.COLOR_RGB2HSV)

    for val in range(0, thimg.shape[ 0 ]):
        now_y = thimg.shape[0]-val-1
        mean_bright = thimg[now_y, :, 2].mean()
        # print(mean_bright)
        if (mean_bright > brightness) & (touch_botten == False):
            down_px = now_y
            # printtxt = "now_y {} ,mean_bright: {}".format(now_y,mean_bright)
            # print(printtxt)
            touch_botten = True

        if touch_botten == True:
            if mean_bright < brightness:
                if diff is True:
                    up_px=now_y
                    # printtxt = "now_y {} ,mean_bright: {}".format(now_y,mean_bright)
                    # print(printtxt)
                    return down_px,up_px
                if diff == False:
                    for val in range(0, thimg.shape[ 0 ]):
                        # print(val)
                        now_y = val +1
                        mean_bright = thimg[now_y, :, 2].mean()
                        if (mean_bright > brightness) :
                            up_px = now_y
                            return down_px,up_px
    #如果亮度判斷一直沒有抓到上下限值, 則輸出None, 並且後面程式會在是否為空
    down_px = None
    up_px = None
    return down_px , up_px

def x_brightness(thimg ,brightness=100, diff=False):
    '''
    diff = True, 亮度偵測會在接觸到第一個符合閥值的px後,繼續往下尋找
    diff = False, 亮度偵測會在接觸到第一個符合閥值的px後,從另外一個邊界尋找

    舉例: y = 0~30, 
    diff= True,30....>28>27>26>......
    diff= False,30....>28>0>1>2>....
    '''
    touch_botten = False

    thimg = cv2.cvtColor(thimg, cv2.COLOR_GRAY2RGB)
    thimg = cv2.cvtColor(thimg, cv2.COLOR_RGB2HSV)
    for val in range(0, thimg.shape[ 1 ]):
        now_x = thimg.shape[1]-val-1
        mean_bright = thimg[:, now_x, 2].mean()
        if (mean_bright > brightness) & (touch_botten == False):
            right_px = now_x
            # printtxt = "now_x {} : {}".format(now_x,mean_bright)
            # print(printtxt)
            touch_botten = True

        if touch_botten == True:
            if mean_bright < brightness:
                if diff is True:
                    left_px=now_x
                    # printtxt = "now_x {} : {}".format(now_x,mean_bright)
                    # print(printtxt)
                    return left_px,right_px
                if diff == False:
                    for val in range(0, thimg.shape[ 1 ]):
                    # print(val)
                        now_x = val+1
                        mean_bright = thimg[:, now_x, 2].mean()
                        if (mean_bright > brightness) :
                            left_px = now_x
                            return left_px,right_px
    #如果亮度判斷一直沒有抓到上下限值, 則輸出None, 並且後面程式會在是否為空
    left_px = None
    right_px = None
    return left_px , right_px

def cut_cropimg(crop_img, file_name, blcbackgroung=blcbackgroung):
    crop_img_shape= crop_img.shape
    blcbackgroung_shape = blcbackgroung.shape
    w = blcbackgroung_shape[1]
    design_range = round(crop_img_shape[1]/(w/2))

    for j in range (0, design_range):
        x2=np.clip(((j + 2)* 160), 0, crop_img_shape[1])

        cut_img= crop_img[:,( j * 160 ): x2]

        cut_img_h,cut_img_w,_=cut_img.shape
        y_loc = int(160- (cut_img_h/2))

        tar_back= blcbackgroung.copy()
        tar_back[ y_loc : y_loc+cut_img_h, 0:cut_img_w,:]=cut_img #create a black image
  
        outputImg_name=(CONV_OUT_PATH+ "/"+ file_name+ "_cut"+ str(j)+ ".jpg")
        cv2.imwrite(outputImg_name, tar_back)

        cutimg_loc_dc[file_name+ "_cut"+ str(j)]=[j * 160, 0, y_loc]
        pass


def get_finger(crop_PCB, filename,lower_fin, upper_fin, pcb_index):
    act_img = crop_PCB.copy()
    h,w,_= act_img.shape
    hsv = cv2.cvtColor(act_img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([10*0.705, 0, 20])#lower_blue
    upper_blue = np.array([30*0.705, 200, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.namedWindow('fingermask1',cv2.WINDOW_NORMAL)
    # cv2.imshow('fingermask1',mask)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(act_img,act_img, mask= mask)

    # mask有很多小白點 嘗試去除
    threshold = h/10 * w/30
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) #计算轮廓所占面积
        if area < threshold:                         #将area小于阈值区域填充黑色
            cv2.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
            continue
    # cv2.namedWindow('fingermask',cv2.WINDOW_NORMAL)
    # cv2.imshow('fingermask',mask)        
    down_px, up_px = y_brightness(mask , 1, diff=False)
    left_px, right_px = x_brightness(mask , 1, diff=False)

    y_extend = 3 
    x_extend = 3
    imgcr=[ up_px - y_extend, 
            down_px + y_extend,
            left_px -x_extend,
            right_px + x_extend]
    # 限制範圍，避免超出圖片原始大小
    imgcr[:1] = np.clip(imgcr[:1], 0 ,h)
    imgcr[2:] = np.clip(imgcr[2:], 0 ,w)
    crop_finger = act_img[ int(imgcr[0]):int(imgcr[1]),
                            int(imgcr[2]):int(imgcr[3]) ] #完整PCB圖片

    h,w = crop_finger.shape[:2]
    if h>w :
        crop_finger = cv2.rotate(crop_finger,cv2.ROTATE_90_CLOCKWISE)
    cv2.namedWindow(str(filename)+ "_"+ str(pcb_index),cv2.WINDOW_NORMAL)
    cv2.imshow(str(filename)+ "_"+ str(pcb_index) ,crop_finger)
    # cv2.waitKey(0)
    blcbackgroung = np.zeros((320, 320, 3), np.uint8)
    cut_cropimg(crop_finger, str(filename)+ "_"+ str(pcb_index), blcbackgroung=blcbackgroung)


def get_pcb(img, filename,lower_pcb, upper_pcb, lower_fin, upper_fin):
    act_img = img.copy()
    h,w,_= act_img.shape
    blcbackgroung = np.zeros((h,w,3), np.uint8)
    hsv = cv2.cvtColor(act_img, cv2.COLOR_BGR2HSV)

    # 70~110 is the range of green in hue
    lower = np.array([lower_pcb*0.705, 40, 20])
    upper = np.array([upper_pcb*0.705, 255, 200])
    mask = cv2.inRange(hsv, lower, upper)# get detect area

    # dilate
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)        
    # cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
    # cv2.imshow("mask",mask)
    
    #將mask跟img做旋轉,且避免直接旋轉造成圖片裁切
    #透過先得到角度,延長轉換後的邊長,進而得到合理的旋轉結果
    # contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # c = sorted(contours, key = cv2.contourArea ,reverse = True)[0]
    # rect = cv2.minAreaRect(c)
    # angle = rect[2]
    # rows, cols = mask.shape[:2]
    # image_center = (cols/2, rows/2)
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

    # abs_cos = abs(M[0,0]) 
    # abs_sin = abs(M[0,1])

    # # find the new width and height bounds
    # bound_w = int(rows * abs_sin + cols * abs_cos)
    # bound_h = int(rows * abs_cos + cols * abs_sin)

    # # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    # M[0, 2] += bound_w/2 - image_center[0]
    # M[1, 2] += bound_h/2 - image_center[1]


    # mask = cv2.warpAffine(mask ,M ,(bound_w, bound_h))
    # act_img = cv2.warpAffine(act_img ,M ,(bound_w, bound_h))

    # cv2.namedWindow("rotate_mask",cv2.WINDOW_NORMAL)
    # cv2.imshow("rotate_mask",rotate_mask)
    # cv2.namedWindow("rotate_img",cv2.WINDOW_NORMAL)
    # cv2.imshow("rotate_img",rotate_img)

    #for video
    if np.all(mask == 0) :
        print("mask return")
        return
    
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt=[] #儲存面積大小的陣列
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) #计算轮廓所占面积
        cnt.append(area)
    sort_cnt= sorted(cnt, reverse=True) #將取出的面積大小排序
    if len(sort_cnt) ==0:
        return 
    tar_area = sort_cnt[0]*0.5 
    
    pcb_index=0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) #计算轮廓所占面积
        if area > tar_area :
            pcb_index +=1

            #將IC區域標記白色
            h,w = mask.shape[:2]
            blcbackgroung = np.zeros((h,w,3), np.uint8)
            draw_label = cv2.drawContours(blcbackgroung.copy(),contours,i,(255,255,255),-1)
           

            #轉成灰階計算二值化
            draw_label_gray = cv2.cvtColor(draw_label.copy(), cv2.COLOR_BGR2GRAY)
            _, label_0ths = cv2.threshold(draw_label_gray, 254, 0, cv2.THRESH_TOZERO)
            # tarb : 目標亮度(跟輸入的pcb大小浮動)
            # label_0ths_shape=label_0ths.shape
            # y_tarb,x_tarb = label_0ths_shape[0]*0.001 ,label_0ths_shape[1]*0.01
         
            down_px, up_px = y_brightness(label_0ths , 1, diff=False)
            left_px, right_px = x_brightness(label_0ths , 1, diff=False)

            # 計算亮度可能因為參數計算, 沒有回傳值, 因此在程式中預先填上None, 並且在後方判斷是否為0
            if (down_px == None) |(up_px == None) | (left_px == None) | (right_px == None) :
                return
            pcb_img = act_img[up_px: down_px, left_px : right_px ] #一個pcb圖片
            
            if np.all(pcb_img == 0) :
                print("crop_PCB return")
                return

            key = str(file_name)+ "_"+ str(pcb_index)
            pcb_loc_dc[key] = [left_px, up_px, right_px, down_px]
            get_finger(pcb_img, filename, lower_fin, upper_fin, pcb_index = pcb_index)
            # cv2.namedWindow(str(pcb_index),cv2.WINDOW_NORMAL)
            # cv2.imshow(str(pcb_index),pcb_img)


#--------------------------find gold finger and cut img--------------------------
for filename in os.listdir(img_path):
    file_name = os.path.splitext(filename)[0]#取得檔名

    img_site = (img_path + "/"+ file_name+ ".jpg")
    print(img_site)
    img = cv2.imread(img_site)   
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.namedWindow(str(filename), cv2.WINDOW_NORMAL)
    # cv2.imshow(str(filename), img.copy())
    get_pcb(img, file_name, 115, 140, 10 ,30)

print(pcb_loc_dc)
print(fin_loc_dc)
cv2.waitKey(0)
cv2.destroyAllWindows

#--------------------------inference--------------------------

# for filename in os.listdir(img_path):
#     file_name = os.path.splitext(filename)[0]#取得檔名

#     img_name = (img_path +"/"+file_name+".jpg")
#     img=cv2.imread(img_name)  #read image 
#     tar_img = img.copy()

#     for idx in range(0, 7):
#         txt_name =txt_path+ "/"+ file_name+ "_cut"+ str(idx)+ ".txt"
#         with open(txt_name, 'r') as f:
#             temp=f.readlines()
#             for line in temp:
#                 cnt= line.strip()#strip()方法用於移除字符串頭尾指定的字符
#                 if (len(cnt)>0) and cnt !='{"bbox":"Not detection."}':
#                     cnt=cnt.split("[")[1]
#                     cnt=cnt.split("]")[0]
#                     # print(cutimg_loc_dc[file_name+ "_cut"+ str(idx)])

#                     infer_loc=[]
#                     for loc_num in cnt.split(","):
#                         infer_loc.append(loc_num)
#                     #print(infer_loc)
#                     #change the element in list to int  
#                     infer_loc=list(map(int,infer_loc))
#                     tar_filename = file_name+ "_cut"+ str(idx)

#                     '''
#                     cutimg_loc_dc[tar_filename][2] means that original img in order to be trained with yolo
#                     therefore we add blackground and change img size to 608*608
#                     so we collect every img's movement and save as cutimg_loc_dc[tar_filename][2]
#                     '''
#                     tarx1 = fin_loc_dc[file_name][0]+ cutimg_loc_dc[tar_filename][0]+ infer_loc[0]+ pcb_x
#                     tary1 = fin_loc_dc[file_name][1]- cutimg_loc_dc[tar_filename][2]+ infer_loc[1]+ pcb_y
#                     tarx2 = fin_loc_dc[file_name][0]+ cutimg_loc_dc[tar_filename][0]+ infer_loc[2]+ pcb_x
#                     tary2 = fin_loc_dc[file_name][1]- cutimg_loc_dc[tar_filename][2]+ infer_loc[3]+ pcb_y
#                     cv2.rectangle(tar_img, (tarx1, tary1), (tarx2, tary2), (0, 255, 0), 2)
#             # print(tarx1, tary1, tarx2, tary2)

# cv2.namedWindow("tar_img",cv2.WINDOW_NORMAL)
# cv2.imshow("tar_img",tar_img)
# # print(cutimg_loc_dc)
# cv2.waitKey(0)
# cv2.destroyAllWindows