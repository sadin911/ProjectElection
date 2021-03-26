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
from tensorflow.keras.applications import MobileNet,InceptionV3,NASNetMobile
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from IPython.display import Image
import glob2
import tensorflow as tf
import datetime
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps
import numpy as np
import cv2
import os
import io
import sys
import shutil
import camscan
path_input = r'images/trainset/*/*.jpg'
path_test = r'images/input/test/*'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('logs',exist_ok=True)
shutil.rmtree('logs')
os.makedirs('images',exist_ok=True)
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

class ScoreClassify:
    def __init__(self):
        self.model = self.createModel()
        self.input_shape = (224,224,3)
        self.model.summary()
        self.pathlist = glob2.glob(path_input)
        self.pathlisttest = glob2.glob(path_test)
        self.path_buffer = []
        self.image_buffer = []
        self.CamScan = camscan.CamScanner(self.input_shape,template='label_form-2.png')
        print(len(self.pathlist))
        for i in range(len(self.pathlist)):
            self.image_buffer.append(open(self.pathlist[i], "rb").read() )
            self.path_buffer.append(self.pathlist[i])
        op = Adam(lr=0.0001)
        self.model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])

    def createModel(self):
        base_model=NASNetMobile( 
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(224,224,3),
            pooling=None,
            classes=8,
            ) 
        
        # x=base_model.output
        # x=Flatten()(x)
        # x=Dense(1024,activation='relu')(x) 
        # x=Dense(1024,activation='relu')(x) 
        # x=Dense(512,activation='relu')(x) 
        # preds=Dense(7,activation='softmax')(x) 
        
        model=Model(inputs=base_model.input,outputs=base_model.output)
        # for layer in model.layers:
        #     layer.trainable=False
        # or if we want to set the first 20 layers of the network to be non-trainabl
        # for layer in model.layers[:20]:
        #     layer.trainable=False
        # for layer in model.layers[20:]:
        #     layer.trainable=True
        
        return model
    
    def gen_data(self,input_byte,input_path):
        img = Image.open(input_byte)
        img = img.resize((self.input_shape[0],self.input_shape[1]))
        # rotate_factor = np.random.normal(loc=1.0, scale=10, size=None)
        # img = img.rotate(rotate_factor)
        flip_flag = np.random.randint(0,1)
        mirror_flag = np.random.randint(0,1)
        if(flip_flag):
            img = ImageOps.flip(img)
        if(mirror_flag):
            img = ImageOps.mirror(img)
            
        blur_rad = np.random.normal(loc=0.0, scale=1.0, size=None)
        img = img.filter(ImageFilter.GaussianBlur(blur_rad))
        
        enhancer_contrat = ImageEnhance.Contrast(img)
        enhancer_brightness = ImageEnhance.Brightness(img)
        contrast_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        brightness_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        img = enhancer_contrat.enhance(contrast_factor)
        img = enhancer_brightness.enhance(brightness_factor)
        # img.save('test_blur.png')
        img = np.asarray(img)/127.5-1
        t = os.path.dirname(input_path)[-1:]
        # print(t,input_path)
        switcher = {
            '0': [1,0,0,0,0,0,0,0],
            '1': [0,1,0,0,0,0,0,0],
            '2': [0,0,1,0,0,0,0,0],
            '3': [0,0,0,1,0,0,0,0],
            '4': [0,0,0,0,1,0,0,0],
            '5': [0,0,0,0,0,1,0,0],
            '6': [0,0,0,0,0,0,0,1],
            '7': [0,0,0,0,0,0,0,1],
            
        }
        target = switcher.get(t)
        # print(t)
        return img,target
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = len(self.pathlist) // batch_size
        permu_ind = list(range(len(self.pathlist)))
        
        step = 0
        for epoch in range(start_epoch,max_epoch):
            permu_ind = np.random.permutation(permu_ind)
            real_index = 0
            for step_index in range(max_step):
                    batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                    batch_target = np.zeros((batch_size,8))
                    for batch_index in range(batch_size):
                        img,target = self.gen_data(io.BytesIO(self.image_buffer[permu_ind[real_index]]),self.path_buffer[permu_ind[real_index]])
                        batch_img[batch_index] = img
                        batch_target[batch_index] = target
                        real_index = real_index+1
                        # print(real_index,len(permu_ind))
                        # self.model.fit(batch_img, batch_target, epochs=1, callbacks=[tensorboard_callback])
                    # print(batch_target)
                    train_loss = self.model.train_on_batch(batch_img,batch_target)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss[0], step=step)
                        tf.summary.scalar('accuracy', train_loss[1], step=step)
                        step = step + 1 
                    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(train_loss))
                    
                    if(step_index%viz_interval==0):
                        rand_index = permu_ind[np.random.randint(0,len(permu_ind))]
                        img,target = self.gen_data(io.BytesIO(self.image_buffer[rand_index]),self.path_buffer[rand_index])
                        img = np.expand_dims(img, axis=0)
                        test_result = self.model.predict(img)
                        # print((test_result),(target))
                        self.test()
                        print(np.argmax(test_result),np.argmax(target))
                        
            os.makedirs('models',exist_ok=True)
            self.model.save('countscoreNASNETMobile_class7.h5')
    def test(self):
        label = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: 'X',
            7: 'X',
        }
       
        i = np.random.randint(0,len(self.pathlisttest))
        input_img = Image.open(self.pathlisttest[i])
        transformed_pil,plot_pil,cropped_data = self.CamScan.cropAll(input_img,0)
        input_shape = self.input_shape
        batch_data = np.zeros((len(cropped_data),input_shape[0],input_shape[1],input_shape[2]))
         
        for index in range(len(cropped_data)):
            batch_data[index,:,:,:]=self.CamScan.prepare_image(cropped_data[index]['image'])
    
    
        result = self.model.predict(batch_data)
        result = np.argmax(result,axis=1)
        result2 = []
        for index in range(len(cropped_data)):
            result2.append(label[result[index]])
        final_img = self.CamScan.plot_result(plot_pil,result2)
        final_img.save(r'images/output_test/'+'test.jpg')
if __name__ == "__main__":
    SC = ScoreClassify()
    # SC.model = load_model('countscoreInception_class8.h5')
    SC.train(1,1000,20,10)

