# -*- coding: utf-8 -*-

import camscan
import glob2
import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.python.keras.models import Model, load_model

path_input = r'images/input/test_v5/*'
pathlist = glob2.glob(path_input)
model = load_model('countscoreInception_class8.h5')
label = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: 'M',
            7: 'X',
        }

for i in range(len(pathlist)):
    input_shape = (299,299,3)
    CamScan = camscan.CamScanner(input_predict=input_shape)
    input_img = Image.open(pathlist[i])
    transformed_pil,plot_pil,cropped_data = CamScan.cropAll(input_img,0)
    
    batch_data = np.zeros((len(cropped_data),input_shape[0],input_shape[1],input_shape[2]))
     
    for index in range(len(cropped_data)):
        batch_data[index,:,:,:]=CamScan.prepare_image(cropped_data[index]['image'])


    result = model.predict(batch_data)
    result = np.argmax(result,axis=1)
    result2 = []
    for index in range(len(cropped_data)):
        result2.append(label[result[index]])
    final_img = CamScan.plot_result(plot_pil,result2)
    final_img.save(r'images/results/'+os.path.basename(pathlist[i]))

