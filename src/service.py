import flask
from flask import Flask
from flask import request, make_response, jsonify
import time
import tensorflow as tf
import base64
import cv2
import json
import io
from PIL import Image
from flask_cors import CORS, cross_origin
import logging
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# from flask_caching import Cache
import os
from camscan import CamScanner
# from memory_profiler import memory_usage

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# limiter = Limiter(
#     app,
#     key_func=get_remote_address,
#     default_limits=["300 per second"]
# )

ec = CamScanner()
crop_col = [1525, 1588, 1651, 1714, 1777, 1840, 1903, 1966, 2029, 2092, 2155, 2218, 2281, 2344, 2407, 2470, 2532, 2596, 2658, 2722,2798]
crop_row = [892, 936, 981, 1026, 1071, 1115, 1160, 1205, 1250, 1294, 1339, 1384, 1429, 1473, 1518, 1563, 1608, 1653, 1697, 1742, 1787, 1832, 1876, 1921, 1966, 2011, 2055, 2100, 2145, 2190, 2235, 2279, 2324, 2369, 2414, 2458, 2503, 2548, 2593, 2637, 2682, 2727, 2772, 2817, 2861, 2906, 2951, 2996, 3040, 3085, 3130, 3175, 3219, 3264, 3309, 3354, 3399]

@app.route('/')
def root():
    return "pyService is running !"


@app.route("/api/election", methods=["POST"])
@cross_origin()
def election():
    if flask.request.method == "POST":
        print('req')
        # print(flask.request.json["image"][0:10])
        start = time.time()
       
        image = Image.open(io.BytesIO(base64.b64decode(flask.request.json["image"]))).convert('RGB')
        end = time.time()
        print(end - start)
    
        data,image_ret = ec.predict(image,crop_col,crop_row)
        
        image_ret.save('wtff2.png')
        # image_ret = image_ret.resize((int(image_ret.size[0]/2),int(image_ret.size[1]/2)))
        ret = {"data":data,"image":"base64 image"}
        
        
        # print(data)
        # print(data)
    return flask.jsonify(ret)



if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    if tf.test.gpu_device_name():
        print('GPU')
    else:
        print("CPU")
    app.run(host = '0.0.0.0',port=8080, threaded=True)
