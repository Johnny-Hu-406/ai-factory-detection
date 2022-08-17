import cv2
import numpy as np

class Defect_detect():

    '''
    There are some method of detecting PCB and gold finger in a image
    '''
    def __init__(self, img, filename):
        self.img = img
        self.filename = filename
        self.pcb_loc_dc = {}
        self.fin_loc_dc = {}
        self.cutimg_loc_dc = {}
    
    # only class and api,in order easy show operation
    def detect(self):

        cut_pcbs = self.get_pcb(lower_green=80, upper_green=110)
        if cut_pcbs is None :
            return "crop_PCB", 0
        else :
            cut_fingers = self.get_finger(cut_pcbs=cut_pcbs,lower_fin=10, upper_fin=35)
            if cut_fingers is None :
                return "finger", 0
            else: 
                loc, cut_list = self.cut_cropimg(cut_fingers=cut_fingers, img_size=320)
                return loc, cut_list

    def y_brightness(self,thimg ,brightness=100, diff=False):
        """
        When brightness detection detect the first px that matches the threshold value, and then continue to search down
        diff = True : continue to search down
        diff = False : search from y=0

        for example: y = 0~30, 
        diff= True,30....>28>27>26>......
        diff= False,30....>28>0>1>2>....
        """
        touch_botten = False
        thimg = cv2.cvtColor(thimg, cv2.COLOR_GRAY2RGB)
        thimg = cv2.cvtColor(thimg, cv2.COLOR_RGB2HSV)

        for val in range(0, thimg.shape[ 0 ]):
            now_y = thimg.shape[0]-val-1
            mean_bright = thimg[now_y, :, 2].mean()

            if (mean_bright > brightness) & (touch_botten == False):
                down_px = now_y
                touch_botten = True

            if touch_botten == True:
                if diff is True:
                    if mean_bright < brightness:
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

        # if not get down_px and up_px, output None
        down_px = None
        up_px = None
        return down_px , up_px

    def x_brightness(self,thimg ,brightness=100, diff=False):
        '''
        When brightness detection detect the first px that matches the threshold value, and then continue to search down
        diff = True : continue to search down
        diff = False : search from x=0

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
                touch_botten = True

            if touch_botten == True:
                if diff is True:
                    if mean_bright < brightness:
                        left_px=now_x
                        # printtxt = "now_x {} : {}".format(now_x,mean_bright)
                        # print(printtxt)
                        return left_px,right_px
                if diff == False:
                    for val in range(0, thimg.shape[ 1 ]):
                    # print(val)
                        now_x = val+1
                        mean_bright = thimg[:, now_x, 2].mean()
                        # print(mean_bright)
                        if (mean_bright > brightness) :
                            left_px = now_x
                            return left_px,right_px
        # if not get down_px and up_px, output None
        left_px = None
        right_px = None
        return left_px , right_px

    def cut_cropimg(self, cut_fingers, img_size):
        img_list = {}
        for cut_finger in cut_fingers:
            tar_img = cut_fingers[cut_finger]
            blcbackgroung = np.zeros((img_size, img_size, 3), np.uint8)
            centerpoint = int(img_size/2)

            tar_img_shape= tar_img.shape
            design_range = round(tar_img_shape[1]/ centerpoint)
            
            for j in range (0, design_range):
                x2=np.clip(((j + 2)* centerpoint), 0, tar_img_shape[1])

                cut_img= tar_img[:,( j * centerpoint ): x2]

                cut_img_h,cut_img_w,_=cut_img.shape
                y_loc = int(centerpoint- (cut_img_h/2))
                
                tar_back= blcbackgroung.copy()
                tar_back[ y_loc : y_loc+cut_img_h, 0:cut_img_w,:]=cut_img #create a black image
                # cv2.namedWindow("tar_back", cv2.WINDOW_NORMAL)
                # cv2.imshow("tar_back",tar_back)
                # cv2.waitKey(0)
                self.cutimg_loc_dc[cut_finger+ "_cut"+ str(j)]=[j * 160, 0, y_loc]
                img_list[cut_finger+ "_cut"+ str(j)]=tar_back

        return [self.pcb_loc_dc, self.fin_loc_dc, self.cutimg_loc_dc], img_list 
        
    def get_finger(self,cut_pcbs, lower_fin, upper_fin):
        cut_fingers={}
        # for idx, cut_pcb in enumerate(cut_pcbs):
        for cut_pcb in cut_pcbs:
            act_img =cut_pcbs[cut_pcb]
            h,w,_= act_img.shape
            hsv = cv2.cvtColor(act_img, cv2.COLOR_BGR2HSV)

            # define range of blue color in HSV
            lower_blue = np.array([lower_fin*0.705, 0, 20])#lower_blue
            upper_blue = np.array([upper_fin*0.705, 200, 255])
            
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # remove noises in mask
            threshold = h/10 * w/30
            contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            #Fill with black on area where are less than the threshold
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i]) 
                if area < threshold:                         
                    cv2.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
                    continue
     
            down_px, up_px = self.y_brightness(mask , brightness=15, diff=False)
            left_px, right_px = self.x_brightness(mask , brightness=5, diff=False)
            if (down_px==None) | (up_px==None) | (left_px==None) | (right_px==None):
                return
            y_extend = 3 
            x_extend = 3
            imgcr=[ up_px - y_extend, 
                    down_px + y_extend,
                    left_px -x_extend,
                    right_px + x_extend]

            imgcr[:1] = np.clip(imgcr[:1], 0 ,h)
            imgcr[2:] = np.clip(imgcr[2:], 0 ,w)

            cut_finger = act_img[ int(imgcr[0]):int(imgcr[1]),
                                    int(imgcr[2]):int(imgcr[3]) ] 
            # cv2.namedWindow(str(cut_pcb),cv2.WINDOW_NORMAL)
            # cv2.imshow(str(cut_pcb),cut_finger)
            
            cut_fingers[cut_pcb] = cut_finger
            self.fin_loc_dc[cut_pcb] = [left_px, up_px, right_px, down_px]
        return cut_fingers
    
    def get_pcb(self,lower_green, upper_green,lower_bright=20, high_bright=200):

        act_img = self.img.copy()
        h,w,_= act_img.shape
        blcbackgroung = np.zeros((h,w,3), np.uint8)
        hsv = cv2.cvtColor(act_img, cv2.COLOR_BGR2HSV)

        lower = np.array([lower_green*0.705, 40, lower_bright])
        upper = np.array([upper_green*0.705, 255, high_bright])
        # get detect area
        mask = cv2.inRange(hsv, lower, upper)
        # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)      
        # cv2.imshow("mask",mask)
        # remove noises in mask
        threshold = h/100 * w/100
        contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i]) 
            if area < threshold:                         
                cv2.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
                continue
        
        # dilate
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)

        # cv2.namedWindow("remove noise mask", cv2.WINDOW_NORMAL)      
        # cv2.imshow("remove noise mask",mask)
        # cv2.waitKey(0)
        
        #for video
        if np.all(mask == 0) :
            return
        
        contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt=[] 
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i]) 
            cnt.append(area)
        sort_cnt= sorted(cnt, reverse=True) 
        if len(sort_cnt) ==0:
            return 
        tar_area = sort_cnt[0]*0.3
        
        pcb_index=0
        cut_pcbs= {}

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i]) 
            if area > tar_area :
                pcb_index +=1

                #Mark target area(PCB) with white
                h,w = mask.shape[:2]
                blcbackgroung = np.zeros((h,w,3), np.uint8)
                draw_label = cv2.drawContours(blcbackgroung.copy(),contours,i,(255,255,255),-1)
                # cv2.namedWindow("draw_label", cv2.WINDOW_NORMAL)      
                # cv2.imshow("draw_label",draw_label)  
                # cv2.waitKey(0)
                draw_label_gray = cv2.cvtColor(draw_label.copy(), cv2.COLOR_BGR2GRAY)
                _, label_0ths = cv2.threshold(draw_label_gray, 254, 0, cv2.THRESH_TOZERO)
                down_px, up_px = self.y_brightness(thimg = label_0ths , brightness=1, diff=False)
                left_px, right_px = self.x_brightness(thimg = label_0ths , brightness=1, diff=False)


                if (down_px == None) |(up_px == None) | (left_px == None) | (right_px == None) :
                    return

                pcb_img = act_img[up_px: down_px, left_px : right_px ] #一個pcb圖片
                cut_pcbs[str(self.filename)+ "_"+ str(pcb_index)] = pcb_img
                # cv2.namedWindow(str(pcb_index),cv2.WINDOW_NORMAL)
                # cv2.imshow(str(pcb_index), pcb_img)
                self.pcb_loc_dc[str(self.filename)+ "_"+ str(pcb_index)] = [left_px, up_px, right_px, down_px]
        return cut_pcbs

def main():
    img_path = "/home/nvidia/Desktop/github/ai-factory-detection/real_situaton_simulation/real_img/LUCID_5.jpg"
    img = cv2.imread(img_path)
    img = np.rot90(img)
    cv2.namedWindow("ori_img", cv2.WINDOW_NORMAL)
    cv2.imshow("ori_img",img)

    defect_detect = Defect_detect(img.copy(), "image")
    defect_detect.detect()

    cv2.waitKey(0)
    cv2.destroyAllWindows

if __name__ == "__main__":
    main()