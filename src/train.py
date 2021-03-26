#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:11:22 2020

@author: trainai
"""

import tensorflow.python.keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D,Flatten
from tensorflow.keras.applications import MobileNet,InceptionV3
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from IPython.display import Image
import glob2
import tensorflow as tf
import datetime
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps,ImageChops
import numpy as np
import cv2
import os
import io
import sys
import shutil
import camscan

from tensorflow.python.keras.layers.merge import add, concatenate
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Input, Dense, Activation , LeakyReLU, Flatten, BatchNormalization, Dropout
from tensorflow.python.keras.layers.recurrent import GRU 
from tensorflow.keras.optimizers import SGD, Adam

path_input = r'images/trainsetALLV6/'
path_test = r'images/input/test_cam/*'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('logs',exist_ok=True)
shutil.rmtree('logs')
os.makedirs('images',exist_ok=True)
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

class ScoreClassify:
    def __init__(self):
        self.input_shape = (64,64,3)
        self.filter_count = 32
        self.kernel_size = (3, 3)
        self.leakrelu_alpha = 0.2
        self.numclass = 7
        self.model = self.createModel()
        self.model.summary()
        self.pathlist = []
        self.bg_buffer = []
        self.bg_path = r'images/trainsetALLV6/*/*.jpg'
        self.pathbglist = glob2.glob(self.bg_path)
        for i in range(len(self.pathbglist)):
            self.bg_buffer.append(open(self.pathbglist[i], "rb").read() )
        self.pad_param = 10
        for i in range(7):
            self.pathlist.append(glob2.glob(path_input+str(i)+'/*.jpg'))
            
        self.train_data_count = [len(i) for i in self.pathlist]
        # self.pathlist = glob2.glob(path_input)    
        self.pathlisttest = glob2.glob(path_test)
        self.path_buffer = []
        self.image_buffer = []
        self.CamScan = camscan.CamScanner()
        # print(self.pathlist)
        print(len(self.pathlist))
        # print(len(self.train_data_count))
        # for i in range(len(self.pathlist)):
        #     self.image_buffer.append(open(self.pathlist[i], "rb").read() )
        #     self.path_buffer.append(self.pathlist[i])
        
        for i in range(len(self.pathlist)):
            temp_image=[]
            temp_path=[]
            for j in range(len(self.pathlist[i])):
                temp_image.append(open(self.pathlist[i][j], "rb").read())
                temp_path.append(self.pathlist[i][j])
            self.image_buffer.append(temp_image)
            self.path_buffer.append(temp_path)
        op = Adam(lr=0.001)
        self.model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])

    def createModel(self):
        base_model=MobileNet(input_shape=(64,64,3),weights=None,include_top=False) 
        x=base_model.output
        x=Flatten()(x)
        x=Dense(1024,activation='relu')(x) 
        x=Dense(512,activation='relu')(x) 
        x=Dense(256,activation='relu')(x) 
        preds=Dense(7,activation='softmax')(x) 
        
        model=Model(inputs=base_model.input,outputs=preds)
        
        return model
    
    def createModel2(self):

        model  = Sequential()
        model.add(Conv2D(self.filter_count, self.kernel_size, input_shape = self.input_shape ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        
        
        model.add(Conv2D(self.filter_count*2, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*2, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*2, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
        model.add(Dropout(0.2))
        
#         model.add(MaxPooling2D(pool_size = (2,2)))
        
#         model.add(Conv2D(self.filter_count*4, self.kernel_size ))
#         model.add(BatchNormalization(momentum = 0.8))
#         model.add(LeakyReLU(alpha = self.leakrelu_alpha))
# #        model.add(Dropout(0.2))
        
#         model.add(Conv2D(self.filter_count*4, self.kernel_size ))
#         model.add(BatchNormalization(momentum = 0.8))
#         model.add(LeakyReLU(alpha = self.leakrelu_alpha))
# #        model.add(Dropout(0.2))
        
#         model.add(Conv2D(self.filter_count*4, self.kernel_size ))
#         model.add(BatchNormalization(momentum = 0.8))
#         model.add(LeakyReLU(alpha = self.leakrelu_alpha))
# #        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Flatten())
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
        
        
        
        model.add(Dense(self.numclass))
        model.add(Activation('softmax'))
        
        
        # op = Adam(lr=0.000001)
        # model.compile(optimizer = op, loss = 'mse' )
        model.summary()

        return model
    
    def pad_and_bg(self,img):
        
        bg_img = np.asarray(Image.open(io.BytesIO( np.random.choice(self.bg_buffer))))
        ret_img = img
        
        
        pad_top = int(abs(np.random.normal(0,self.pad_param)))
        pad_bottom = int(abs(np.random.normal(0,self.pad_param)))
        pad_left = int(abs(np.random.normal(0,self.pad_param)))
        pad_right = int(abs(np.random.normal(0,self.pad_param)))
        
        
        # trim_top = int(abs(np.random.normal(0,self.trim_param)))
        # trim_bottom = int(abs(np.random.normal(0,self.trim_param)))
        # trim_left = int(abs(np.random.normal(0,self.trim_param)))
        # trim_right = int(abs(np.random.normal(0,self.trim_param)))
        
        ret_img = cv2.copyMakeBorder( np.asarray(ret_img), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        
        #mask
        mask_img = np.zeros((img.size[0],img.size[1],3))
        # mask_2 = Image.fromarray(mask_img.astype('uint8')).rotate(rotate_param,resample = Image.NEAREST,expand = True, fillcolor = (255,255,255))
        mask_3 = cv2.copyMakeBorder( np.asarray(mask_img), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        
        ret_img = cv2.resize(ret_img, dsize=(img.size[0], img.size[1]))
        bg_img = cv2.resize(bg_img, dsize=(img.size[0], img.size[1]))
        mask_3 = cv2.resize(mask_3, dsize=(img.size[0], img.size[1]))
        
        mask_3 = mask_3/255
        ret_img = ret_img*(1-mask_3) + bg_img*mask_3
        ret_img = Image.fromarray(ret_img.astype('uint8'))
        
        return  ret_img
    
    def gen_data2(self,input_byte,input_path):
        img = Image.open(input_byte)
        img = img.resize((self.input_shape[0],self.input_shape[1]))
        
        img = self.pad_and_bg(img)
        
        blur_rad = np.random.normal(loc=0.0, scale=2, size=None)
        img = img.filter(ImageFilter.GaussianBlur(blur_rad))
       
        enhancer_contrat = ImageEnhance.Contrast(img)
        enhancer_color = ImageEnhance.Color(img)
        enhancer_brightness = ImageEnhance.Brightness(img)
        contrast_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        brightness_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        color_factor = np.max([0,1-abs(np.random.normal(loc=0, scale=0.5, size=None))])
        
        img = enhancer_contrat.enhance(contrast_factor)
        img = enhancer_brightness.enhance(brightness_factor)
        img = enhancer_color.enhance(color_factor)
        # img.save('test_blur.png')
        img = np.asarray(img)/127.5-1
        t = os.path.dirname(input_path)[-1:]
        # print(t,input_path)
        switcher = {
            '1': [1,0,0,0,0,0,0],
            '2': [0,1,0,0,0,0,0],
            '3': [0,0,1,0,0,0,0],
            '4': [0,0,0,1,0,0,0],
            '5': [0,0,0,0,1,0,0],
            '6': [0,0,0,0,0,1,0],
            '0': [0,0,0,0,0,0,1],
        }
        target = switcher.get(t)
        # print(t)
        return img,target
    
    def gen_data(self,input_byte,input_path):
        img = Image.open(input_byte)
        img = img.resize((self.input_shape[0],self.input_shape[1]))
        # img = self.pad_and_bg(img)
        # rotate_factor = np.random.normal(loc=1.0, scale=10, size=None)
        # img = img.rotate(rotate_factor)
        # flip_flag = np.random.randint(0,1)
        # mirror_flag = np.random.randint(0,1)
        # gray_flag = np.random.randint(0,1)
        # if(gray_flag):
        #     img = img.convert('LA')
        # if(flip_flag):
        #     img = ImageOps.flip(img)
        # if(mirror_flag):
        #     img = ImageOps.mirror(img)
            
        blur_rad = np.random.normal(loc=0.0, scale=2, size=None)
        img = img.filter(ImageFilter.GaussianBlur(blur_rad))
        
        enhancer_contrat = ImageEnhance.Contrast(img)
        enhancer_brightness = ImageEnhance.Brightness(img)
        enhancer_color = ImageEnhance.Color(img)
        contrast_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        color_factor = np.max([0,1-abs(np.random.normal(loc=0, scale=0.5, size=None))])

        rotate_factor = np.random.normal(loc=0, scale=0.17, size=None)
        translate_factor_hor = np.random.normal(loc=0, scale=5, size=None)
        translate_factor_ver = np.random.normal(loc=0, scale=5, size=None)
        brightness_factor = np.random.normal(loc=1.0, scale=0.5, size=None)

        img = enhancer_contrat.enhance(contrast_factor)
        img = enhancer_brightness.enhance(brightness_factor)
        img = enhancer_color.enhance(color_factor)
        img = ImageChops.offset(img, int(translate_factor_hor), int(translate_factor_ver))
        img = img.rotate(np.rad2deg(rotate_factor))
        
        img.save('test_blur.png')
        img = np.asarray(img)/127.5-1
        t = os.path.dirname(input_path)[-1:]
        # print(t,input_path)
        switcher = {
            '0': [1,0,0,0,0,0,0],
            '1': [0,1,0,0,0,0,0],
            '2': [0,0,1,0,0,0,0],
            '3': [0,0,0,1,0,0,0],
            '4': [0,0,0,0,1,0,0],
            '5': [0,0,0,0,0,1,0],
            '6': [0,0,0,0,0,0,1],
           
        
            
        }
        target = switcher.get(t)
        # print(t)
        return img,target
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = sum([len(i) for i in self.pathlist]) // batch_size
        # permu_ind = list(range(len(self.pathlist)))
        
        step = 0
        for epoch in range(start_epoch,max_epoch):
            # permu_ind = np.random.permutation(permu_ind)
            real_index = 0
            for step_index in range(max_step):
                    batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                    batch_target = np.zeros((batch_size,7))
                    
                    for batch_index in range(batch_size):
                        tget = batch_index % self.numclass
                        random_index = np.random.randint(self.train_data_count[tget])
                        img,target = self.gen_data(io.BytesIO(self.image_buffer[tget][random_index]),self.path_buffer[tget][random_index])
                        batch_img[batch_index] = img
                        batch_target[batch_index] = target
                        real_index = real_index+1
                    save_img = (batch_img[np.random.randint(batch_size)]+1)*127.5
                    save_img = Image.fromarray(save_img.astype('uint8'))
                    save_img.save('temppp.png')
                    # print(batch_target)
                    train_loss = self.model.train_on_batch(batch_img,batch_target)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss[0], step=step)
                        tf.summary.scalar('accuracy', train_loss[1], step=step)
                        step = step + 1 
                    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(train_loss))
                    
                    # if(step_index%viz_interval==0):
                    # #     rand_index = permu_ind[np.random.randint(0,len(permu_ind))]
                    # #     img,target = self.gen_data(io.BytesIO(self.image_buffer[rand_index]),self.path_buffer[rand_index])
                    # #     img = np.expand_dims(img, axis=0)
                    # #     test_result = self.model.predict(img)
                    # #     # print((test_result),(target))
            self.test()
                    # #     print(np.argmax(test_result),np.argmax(target))
                        
            os.makedirs('models',exist_ok=True)
            self.model.save('countscoreMobile_64_class7_tv6_transAll_rotate.h5')
            
    def test(self):
        #template v6
        crop_col = [1525, 1588, 1651, 1714, 1777, 1840, 1903, 1966, 2029, 2092, 2155, 2218, 2281, 2344, 2407, 2470, 2532, 2596, 2658, 2722,2798]
        crop_row = [892, 936, 981, 1026, 1071, 1115, 1160, 1205, 1250, 1294, 1339, 1384, 1429, 1473, 1518, 1563, 1608, 1653, 1697, 1742, 1787, 1832, 1876, 1921, 1966, 2011, 2055, 2100, 2145, 2190, 2235, 2279, 2324, 2369, 2414, 2458, 2503, 2548, 2593, 2637, 2682, 2727, 2772, 2817, 2861, 2906, 2951, 2996, 3040, 3085, 3130, 3175, 3219, 3264, 3309, 3354, 3399]
        label = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: 'X',
        }
       
        i = np.random.randint(0,len(self.pathlisttest))
        input_img = Image.open(self.pathlisttest[i])
        transformed_pil,plot_pil,cropped_data = self.CamScan.cropAll(input_img,crop_col,crop_row,10)
        input_shape = self.input_shape
        batch_data = np.zeros((len(cropped_data),input_shape[0],input_shape[1],input_shape[2]))
         
        for index in range(len(cropped_data)):
            batch_data[index,:,:,:]=np.asarray(cropped_data[index]['image'].resize((self.input_shape[0],self.input_shape[1])))
    
        batch_data = (batch_data/127.5)-1
        result = self.model.predict(batch_data)
        result = np.argmax(result,axis=1)
        result2 = []
        for index in range(len(cropped_data)):
            result2.append(label[result[index]])
        final_img = self.CamScan.plot_result(plot_pil,result2,crop_col,crop_row)
        final_img.save(r'images/output_test/'+ os.path.basename(self.pathlisttest[i]))
if __name__ == "__main__":
    SC = ScoreClassify()
    # SC.model = load_model('countscoreMobile_64_class7_tv6_transAll_rotate.h5')
    SC.train(1,10000,32,10)

