# !pip install requests
import numpy as np
import pandas as pd
import random
from flask import Flask, request, render_template, Response
from jinja2 import Template
from skimage.io import imread, imsave
import glob
from predict import predict_pose,image_processing,get_model
from importlib import import_module
import os
import requests
import flask
from io import BytesIO
from PIL import Image
from skimage.color import rgb2gray
from tensorflow.keras.models import load_model

model = load_model('model/best_model/model.h5')

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

app = Flask(__name__)

@app.route('/')
def index():
    #Goes to html
    return render_template('index.html')
#
# app.config["image_upload"] ='C://Users//m_mas//Desktop//Bitmaker_DSI//submit-sara//capstone//static//images//upload//'

@app.route('/uploadpicture', methods=['POST','GET'])
def picture():
    if request.method == 'POST':
        result = request.form
            # return redirect(request.url)

        #Code for image processing goes here
        return render_template('upload_pic.html')

@app.route('/video_prediction', methods=['POST','GET'])
def video():
    if request.method == 'POST':
        result = request.form
    return render_template('video_prediction.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/upload_url', methods=['POST','GET'])
# def video():
#     if request.method == 'POST':
#         result = request.form
#
#     return render_template('upload_url.html')

@app.route('/result_image', methods=['POST','GET'])
def predict():

    if request.method == 'POST':
       result = flask.request.form
    url = result['submit']
    response = requests.get(url)
    print(response)
    img = Image.open(BytesIO(response.content))

    img = img.resize((200,200)).convert(mode='L')
    img = np.array(img, dtype = "float32")

    pred = model.predict(np.reshape(img,(1,200,200,1)))
    pred = np.argmax(model.predict(img.reshape(1,200,200,1)))

    value = np.max(pred)
    value = round(value * 100, 2)

    classes = ['Downward Dog', 'Tree', 'Triangle', 'Upward Dog', 'Warrior 1', 'Warrior 2']
    yoga_pose = classes[pred]

    # yoga_pose, value = predict_pose(img)

    return render_template('result_image.html', yoga_pose = yoga_pose, value = value)


if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT =  4085
    app.run(HOST, PORT,debug = True)
