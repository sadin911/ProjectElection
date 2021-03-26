#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:08:18 2020

@author: chonlatid
"""

from PIL import Image,ImageDraw, ImageFont

from tensorflow.python.keras.models import load_model

import numpy as np

import cv2
import time


class CamScanner():
    def __init__(self,input_predict=(299,299,3),template='template_v5.png'):
       
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.gf = 16
        self.input_predict = input_predict
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
        self.predict_input_shape = (64,64,3)
        self.model = load_model('corner.h5')
        self.emodel = load_model('demo2.h5')
        self.label = {
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '-1',
            0: '0',
        }
        self.row_per_page = 7
        self.numfeature =50000
        self.detector = cv2.QRCodeDetector()
        self.template_pil = Image.open('template_v6.png').convert('RGB')
        self.warp = []
        # self.col_edge=[ 353,  568,  763,  832, 1930, 2089]
        # self.row_edge=[ 355,  486,  682,  873,  911,  948,  988, 1026]
        
        # self.crop_col = [759, 790, 821, 853, 884, 915, 947, 978, 1009, 1041, 1072, 1104, 1135, 1166, 1198, 1229, 1260, 1292, 1323, 1355,1393]
        # self.crop_row = [443, 465, 487, 510, 532, 554, 577, 599, 621, 644, 666, 688, 711, 733, 756, 778, 800, 823, 845, 867, 890, 912, 934, 957, 979, 1001, 1024, 1046, 1069, 1091, 1113, 1136, 1158, 1180, 1203, 1225, 1247, 1270, 1292, 1314, 1337, 1359, 1382, 1404, 1426, 1449, 1471, 1493, 1516, 1538, 1560, 1583, 1605, 1627, 1650, 1672, 1695]
        
        #template v6
        # self.crop_col = [1525, 1588, 1651, 1714, 1777, 1840, 1903, 1966, 2029, 2092, 2155, 2218, 2281, 2344, 2407, 2470, 2532, 2596, 2658, 2722,2798]
        # self.crop_row = [892, 936, 981, 1026, 1071, 1115, 1160, 1205, 1250, 1294, 1339, 1384, 1429, 1473, 1518, 1563, 1608, 1653, 1697, 1742, 1787, 1832, 1876, 1921, 1966, 2011, 2055, 2100, 2145, 2190, 2235, 2279, 2324, 2369, 2414, 2458, 2503, 2548, 2593, 2637, 2682, 2727, 2772, 2817, 2861, 2906, 2951, 2996, 3040, 3085, 3130, 3175, 3219, 3264, 3309, 3354, 3399]
        
        #template_crop
        
        
        
        #train_template
        # self.crop_col = [101, 211, 321, 432, 542, 653, 763, 873, 984, 1094, 1205, 1315, 1425, 1536, 1646, 1757, 1867, 1978]
        # self.crop_row = [312, 393, 474, 555, 636, 717, 799, 880, 961, 1042, 1123, 1205]
        
    def cropAll(self,image_pil,crop_col,crop_row,extend=0,randomshift_x=0,randomshift_y=0):
        
        template_pil = self.template_pil.resize((self.template_pil.width//1,self.template_pil.height//1))
        # self.warp = self.predictcorner(image_pil)
        self.warp = image_pil.convert('RGB')
        # warped_re = self.warp.resize((self.warp.width//1,self.warp.height//1))
        warped_re = image_pil.convert('RGB')
        
        try:
            print('trying qr')
            transformed_pil = self.templateAlignmentbyQR(warped_re,template_pil)
            
        except:
            print("cannot read qr")
            transformed_pil = self.templateAlignment(warped_re,template_pil,False)
        
        
        plot_pil = self.draw_line(transformed_pil,crop_row,crop_col)
        cropped_data = self.crop_data(transformed_pil,crop_col,crop_row,extend,randomshift_x,randomshift_y)
        return transformed_pil,plot_pil,cropped_data
    
    
    def templateAlignmentbyQR(self,image_pil,template_pil):
        template_color = np.asarray(template_pil)
        image_color = np.asarray(image_pil)
        # warp_color = np.asarray(self.warp)
        img1 = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) 
        img2 = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY) 
        # img_warp = cv2.cvtColor(warp_color, cv2.COLOR_BGR2GRAY) 
        height, width = img2.shape 
        height_im1, width_im1 = img1.shape 
        
       
            
        # img1 = self.mask_qr_only(img1).astype('uint8')
        # img2 = self.mask_qr_only(img2).astype('uint8')
        
        # kp1, d1 = orb_detector.detectAndCompute(img1, None) 
        # kp2, d2 = orb_detector.detectAndCompute(img2, None) 
          
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
          
        # matches = matcher.match(d1, d2) 
          
        # matches.sort(key = lambda x: x.distance) 
          
        # matches = matches[:int(len(matches)*10)] 
        # no_of_matches = len(matches) 
          
 
        # p1 = np.zeros((no_of_matches, 2)) 
        # p2 = np.zeros((no_of_matches, 2)) 
          
        # for i in range(len(matches)): 
        #   p1[i, :] = kp1[matches[i].queryIdx].pt 
        #   p2[i, :] = kp2[matches[i].trainIdx].pt 
          
        retval1, decoded_info1, points1, straight_qrcode1 = self.detector.detectAndDecodeMulti(image_color)
        retval2, decoded_info2, points2, straight_qrcode2 = self.detector.detectAndDecodeMulti(template_color)
        print(decoded_info1)
        align_point1=[*points1[decoded_info1.index('{TopLeft}')],
                     *points1[decoded_info1.index('{TopRight}')],
                     *points1[decoded_info1.index('{BotRight}')],
                     *points1[decoded_info1.index('{BotLeft}')]
                     ]
        align_point2=[*points2[decoded_info2.index('{TopLeft}')],
                     *points2[decoded_info2.index('{TopRight}')],
                     *points2[decoded_info2.index('{BotRight}')],
                     *points2[decoded_info2.index('{BotLeft}')]
                     ]

        homography, mask = cv2.findHomography(np.array(align_point1), np.array(align_point2), cv2.RANSAC) 
        # img_plot = cv2.drawKeypoints(image_color, kp1, None,color=(255,0,0),flags=0)
        # cv2.imwrite('test2.jpg', img_plot) 
        transformed_img = cv2.warpPerspective(image_color, 
                            homography, (width, height))
        transformed_pil = Image.fromarray(transformed_img.astype('uint8'))
        
        return transformed_pil
    def templateAlignment(self,image_pil,template_pil,crop_mode=False):
        template_color = np.asarray(template_pil)
        image_color = np.asarray(image_pil)
        img1 = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) 
        img2 = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY) 
        height, width = img2.shape 
        height_im1, width_im1 = img1.shape 
        
        if(crop_mode):
            wt0 = (int)(width/37) 
            ht0 = (int)(height/5.5)
            wt1 = (int)(width - wt0)
            ht1 = (int)(height -1.4*ht0)
            img2 = cv2.rectangle(img2,(wt0,ht0),(wt1,ht1),(255,255,255,0),thickness=cv2.FILLED)
            wt0_im1 = (int)(width_im1/37) 
            ht0_im1 = (int)(height_im1/5)
            wt1_im1 = (int)(width_im1 - wt0_im1)
            ht1_im1 = (int)(height_im1 - 1.4*ht0_im1)
            img1 = cv2.rectangle(img1,(wt0_im1,ht0_im1),(wt1_im1,ht1_im1),(255,255,255,0),thickness=cv2.FILLED)
            print(img2.shape)
            print(ht1,wt1)
            cv2.imwrite('testcrop.jpg', img2)
            cv2.imwrite('testcrop_img1.jpg', img1)
            
        orb_detector = cv2.ORB_create(self.numfeature) 

        img1 = self.mask_qr_only(img1).astype('uint8')
        img2 = self.mask_qr_only(img2).astype('uint8')
        
        kp1, d1 = orb_detector.detectAndCompute(img1, None) 
        kp2, d2 = orb_detector.detectAndCompute(img2, None) 
          
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
          
        matches = matcher.match(d1, d2) 
          
        matches.sort(key = lambda x: x.distance) 
          
        matches = matches[:int(len(matches)*10)] 
        no_of_matches = len(matches) 
          
 
        p1 = np.zeros((no_of_matches, 2)) 
        p2 = np.zeros((no_of_matches, 2)) 
          
        for i in range(len(matches)): 
          p1[i, :] = kp1[matches[i].queryIdx].pt 
          p2[i, :] = kp2[matches[i].trainIdx].pt 
          
          

        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
        # img_plot = cv2.drawKeypoints(image_color, kp1, None,color=(255,0,0),flags=0)
        # cv2.imwrite('test2.jpg', img_plot) 
        transformed_img = cv2.warpPerspective(image_color, 
                            homography, (width, height))
        transformed_pil = Image.fromarray(transformed_img.astype('uint8'))
        
        return transformed_pil
    
    def predictcorner(self,img_viz):
        ratio_h = img_viz.height / self.img_rows
        ratio_w = img_viz.width / self.img_cols
        orgimg = np.asarray(img_viz)
        orgimg = orgimg/127.5 - 1
        
        img_viz = img_viz.resize((256,256))
        img_viz = np.asarray(img_viz)
        img_viz = img_viz/127.5 - 1
        
        indput_data = img_viz
        indput_data = np.expand_dims(indput_data, axis = 0)
        
        [predict_mask,predict_corner] = self.model.predict(indput_data)
        predict_corner = np.reshape(predict_corner[0],(4,2))
        predict_corner[:,0] *= ratio_w  
        predict_corner[:,1] *= ratio_h
        predict_corner = np.reshape(predict_corner,(8))
        
        predict_mask = cv2.resize(predict_mask[0],(orgimg.shape[1],orgimg.shape[0]))
        warped = self.transformFourPoints((orgimg+1)*127.5, predict_corner.reshape(4, 2))
        warped = Image.fromarray(warped.astype('uint8'),'RGB')
        return warped

    def onlycorner(self,img_viz,width,height):
        ratio_h = height / self.img_rows
        ratio_w = width / self.img_cols
        orgimg = np.asarray(img_viz).astype('float')
        orgimg = orgimg/127.5 - 1

        img_viz = np.asarray(img_viz)
        img_viz = img_viz/127.5 - 1
        
        indput_data = img_viz
        indput_data = np.expand_dims(indput_data, axis = 0)
        
        [predict_mask,predict_corner] = self.model.predict(indput_data)
        predict_corner = np.reshape(predict_corner[0],(4,2))
        predict_corner[:,0] *= ratio_w  
        predict_corner[:,1] *= ratio_h
        predict_corner = np.reshape(predict_corner,(8))

        return str(predict_corner)

    def order_points(self,pts):
    	rect = np.zeros((4, 2), dtype="float32")
    	s = pts.sum(axis=1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    	diff = np.diff(pts, axis=1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
    	return rect

    def transformFourPoints(self,image_cv, pts):

        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
    
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_cv, M, (maxWidth, maxHeight))
    
        return warped
    
    def draw_line(self,image,crop_row,crop_col):
        image = image.convert('RGB')
        img1 = ImageDraw.Draw(image)  
        for i in range(len(crop_col)):
            img1.line([(crop_col[i],crop_row[0]),(crop_col[i],crop_row[-1])], fill =(0,255,255), width = 1) 
        for i in range(len(crop_row)):
            img1.line([(crop_col[0],crop_row[i]),(crop_col[-1],crop_row[i])], fill =(0,255,255), width = 1) 
        
        return image
    
    def plot_result(self,image,result,crop_col,crop_row):
        image = image.convert('RGB')
        img1 = ImageDraw.Draw(image)  
        fnt = ImageFont.truetype("FreeMono.ttf",size=60)
        index=0
        for row_index in range(len(crop_row)-1):
             for col_index in range(len(crop_col)-1):
                 img1.text((crop_col[col_index]+10,crop_row[row_index]), result[index], font=fnt, fill=(200,0,255))
                 index=index+1
        return image
            
    def crop_data(self,image,crop_col,crop_row,extend=0,randomshift_x=0,randomshift_y=0):
        image = image.convert('RGB')
        index=0
        result=[]

        for row_index in range(len(crop_row)-1):
             for col_index in range(len(crop_col)-1):
                 x= int(np.random.normal(0,randomshift_x))
                 y= int(np.random.normal(0,randomshift_y))
                 result.append({
                     "image":image.crop(( crop_col[col_index]-extend+x, crop_row[row_index]-extend+y,crop_col[col_index+1]+extend+x,  crop_row[row_index+1]+extend +y )),
                     "index":index,
                     "location":(crop_col[col_index],crop_row[row_index],crop_col[col_index+1], crop_row[row_index+1])
                     })
        return result
    
    def prepare_image(self,img):
        img = img.resize((self.input_predict[0],self.input_predict[1]))
        img = np.asarray(img)/127.5-1
        img = np.expand_dims(img, axis=0)
        return img

    def mask_qr_only(self, img):
        detector = cv2.QRCodeDetector()
        retval, points = detector.detectMulti(img)
        mask = np.zeros(img.shape,np.uint8)
        cv2.fillPoly(mask, points.astype(int),1)
        img = 255*(mask==0) + np.multiply(mask,img)
        return img
    
    def predict(self,input_img,crop_col,crop_row):
        # input_img.save('req.png')
        # input_img = Image.open(self.pathlisttest[i])
        start = time.time()
        print("start crop all")
        transformed_pil,plot_pil,cropped_data = self.cropAll(input_img,crop_col,crop_row,10)
        end = time.time()
        print(end - start)
        input_shape = self.predict_input_shape
        batch_data = np.zeros((len(cropped_data),input_shape[0],input_shape[1],input_shape[2]))
         
        start = time.time()
        print("start resize")
        for index in range(len(cropped_data)):
            batch_data[index,:,:,:]=np.asarray(cropped_data[index]['image'].resize((input_shape[0],input_shape[1])))
        end = time.time()
        print(end - start)   
        batch_data = (batch_data/127.5)-1
        

        start = time.time()
        print("start predict")
        result = self.emodel.predict(batch_data)
        end = time.time()
        print(end - start)
        
        start = time.time()
        print("start answer")
        result = np.argmax(result,axis=1)
        result2 = []
        for index in range(len(cropped_data)):
            result2.append(self.label[result[index]])
        
        ret = []
        num = len(cropped_data)//self.row_per_page
        for i in range(self.row_per_page):
            ret.append({"rawscore":[]})
        for i in range(len(cropped_data)):
            ret[i//num]["rawscore"].append(int(result2[i] ))
        img_result = self.plot_result(plot_pil,result2,crop_col,crop_row)
        for i in range(len(ret)):
            ret[i]["score"]=int(np.sum( np.where( np.array(ret[i]["rawscore"])==-1,0, np.array(ret[i]["rawscore"]))))
        
        for i in range(self.row_per_page):
            ret[i]["corrected_raw"] = self.correct_score(ret[i]["rawscore"])
        for i in range(len(ret)):
            ret[i]["corrected_score"]=int(np.sum( np.where( np.array(ret[i]["corrected_raw"])==-1,0, np.array(ret[i]["corrected_raw"]))))
          
        end = time.time()
        print(end - start)
        return ret,img_result

    
    def correct_score(self,inn):
        null_mask = np.where(np.array(inn)==-1,1,0)
        leng = len(inn)
        sumleft = [0]
        sumright = [0]
        for i in range(1,len(inn)):
            sumleft.append(sumleft[-1]+ (inn[i-1] if inn[i-1]!=-1 else 0))
            sumright.insert(0,sumright[0]+(inn[leng - i] if inn[leng - i]!=-1 else 0))
            
        # i5 = [5*i for i in range(leng)]
        
        
        i5_mask = np.where(np.array(inn)==-1,0,5)
        i5 = [0]
        for i in range(1,len(inn)):
            i5.append(i5[-1]+ (i5_mask[i-1] if i5_mask[i-1]!=-1 else 0))
        
        #        error left               + error right
        error = (i5-np.array(sumleft)) + np.array(sumright) #compare with ideal [5,5,5,5,5,x,0,0,0,0,0]
        
        # print(sumleft)
        # print(sumright)
        # print(error)
        
        indexth = np.argmin(error)
        
        b1=[5 for i in range(indexth)]
        b2=[0 for i in range(leng-indexth-1)]
        
        result = [*b1,inn[indexth] if inn[indexth] !=-1 else 0,*b2]
        result = np.where(np.array(null_mask)==1,-1,result).tolist()
        return result
if __name__ == "__main__":
    # from PIL import Image
    image = Image.open(r"IMG_1624.png")

    # ec = econfig(image,[[832,915],[3136,902],[3148,1840],[824,1831]]) #w/h => 4032/3024
    ec = CamScanner() #w/h => 4032/3024
    pred,img_result = ec.predict(image)
    # img_result.save('wtf.png')
    # res,img_result = ec.plot_result(image, result)
    # label=[]
    # for i in range(1120):
    #     label.append(str(i))
    # image2 = ec.plot_result(image, label)
    
    # image2.save('temp.png')
    # transformed_pil,plot_pil,cropped_data = ec.cropAll(image,0)
    # image3 = ec.draw_line(transformed_pil)
    # image3.save('ss/temp2.png')
    # ec.image.show()
    # ec.draw_line(image).save('djs.png')
    #ref point 304,303   2149,303   2149,1039 304,1039 w/h =>2339,1653
    # data = ec.crop_data(image,0)
    # for i in range(len(data)):
    #     data[i]['image'].save('/media/trainai/data/db/election/Score/train/0010000/'+str(i)+'a.png')
