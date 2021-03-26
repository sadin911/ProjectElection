
import camscan
import glob2
import os
import shutil
from PIL import Image

path_input = r'images/input/train_v6/*.jpg'
path_template = r'template_v6.png'
pathlist = glob2.glob(path_input)
crop_col = [1525, 1588, 1651, 1714, 1777, 1840, 1903, 1966, 2029, 2092, 2155, 2218, 2281, 2344, 2407, 2470, 2532, 2596, 2658, 2722,2798]
crop_row = [892, 936, 981, 1026, 1071, 1115, 1160, 1205, 1250, 1294, 1339, 1384, 1429, 1473, 1518, 1563, 1608, 1653, 1697, 1742, 1787, 1832, 1876, 1921, 1966, 2011, 2055, 2100, 2145, 2190, 2235, 2279, 2324, 2369, 2414, 2458, 2503, 2548, 2593, 2637, 2682, 2727, 2772, 2817, 2861, 2906, 2951, 2996, 3040, 3085, 3130, 3175, 3219, 3264, 3309, 3354, 3399]
CamScan = camscan.CamScanner(path_template,template=path_template)
for i in range(len(pathlist)):
    crop_row = [[3040, 3085, 3130, 3175, 3219, 3264, 3309, 3354],
                [892, 936, 981, 1026, 1071, 1115, 1160, 1205],
                [1250, 1294, 1339, 1384, 1429, 1473, 1518, 1563],
                [1608, 1653, 1697, 1742, 1787, 1832, 1876, 1921],
                [1966, 2011, 2055, 2100, 2145, 2190, 2235, 2279],
                [2324, 2369, 2414, 2458, 2503, 2548, 2593, 2637],
                [2682, 2727, 2772, 2817, 2861, 2906, 2951, 2996],
                ]
    for j in range(7):
        for m in [10]:
           
            input_img = Image.open(pathlist[i])
            transformed_pil,plot_pil,cropped_data = CamScan.cropAll(input_img,crop_col,crop_row[j],m)
            # shutil.rmtree('images/output')
            os.makedirs('images',exist_ok=True)
            os.makedirs('images/output/out_aligned',exist_ok=True)
            os.makedirs('images/output/out_plot',exist_ok=True)
            os.makedirs('images/trainsetALLV6/'+ str(j) ,exist_ok=True)
            
            save_aligned = os.path.join('images','output','out_aligned', os.path.basename(pathlist[i])[0:-4] + '.jpg')
            save_plot = os.path.join('images','output','out_plot', os.path.basename(pathlist[i])[0:-4] + '.jpg')
            
            for n in range(len(cropped_data)):
                save_crop = os.path.join('images','trainsetALLV6',str(j), str(n) + '_' + str(m) + '_' + str(i) + '.jpg')
                cropped_data[n]['image'].save(save_crop)
                
            transformed_pil.save(save_aligned)
            plot_pil.save(save_plot)



