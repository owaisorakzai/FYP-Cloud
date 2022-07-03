from flask import Flask
from flask import request, jsonify
from random import sample
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
from skimage import transform
from PIL import Image
import os

server = Flask(__name__)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def run_request():
    index = int(request.json['index'])
    list = ['red', 'green', 'blue', 'yellow', 'black']
    return list[index]

@server.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            model = load_model('fypModel.h5')
            img=load(file_path)
            index = np.argmax(model.predict(img), axis=-1)
            print(index[0])
            classes=['Normal','Good','Bad']
            return classes[index[0]]
    else:
        return run_request()

