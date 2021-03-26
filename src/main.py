
import camscan
import glob2
import os
import shutil
from PIL import Image

path_input = r'images/input/test_A0/*jpg'
path_template = r'template_v5_blank.png'
pathlist = glob2.glob(path_input)
CamScan = camscan.CamScanner(path_template,template=path_template)
for i in range(len(pathlist)):
   
    input_img = Image.open(pathlist[i])
    transformed_pil,plot_pil,cropped_data = CamScan.cropAll(input_img,10)
     

    # shutil.rmtree('images/output')
    os.makedirs('images',exist_ok=True)
    os.makedirs('images/output/out_aligned',exist_ok=True)
    os.makedirs('images/output/out_plot',exist_ok=True)
    os.makedirs('images/trainset/'+ os.path.basename(pathlist[i])[0] ,exist_ok=True)
    
    save_aligned = os.path.join('images','output','out_aligned', os.path.basename(pathlist[i])[0:-4] + '.jpg')
    save_plot = os.path.join('images','output','out_plot', os.path.basename(pathlist[i])[0:-4] + '.jpg')
    
    for n in range(len(cropped_data)):
        save_crop = os.path.join('images','trainset',os.path.basename(pathlist[i])[0], str(n) + '@10*' + os.path.basename(pathlist[i])[-5] + '.jpg')
        cropped_data[n]['image'].save(save_crop)
        
    transformed_pil.save(save_aligned)
    plot_pil.save(save_plot)



