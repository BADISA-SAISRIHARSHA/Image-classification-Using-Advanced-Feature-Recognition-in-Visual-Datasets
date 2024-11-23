from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model

from PIL import Image as im
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

import sqlite3

import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage


import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

from torchvision.models import detection
import sqlite3
import torch
from torchvision import models
from flask import Flask, render_template, request, redirect, Response

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


app = Flask(__name__)


from torchvision.models import detection
import sqlite3
import torch
from torchvision import models  
from ultralytics import YOLO
from flask import request, Flask, jsonify

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path1 = 'model_bin.h5' # load .h5 Model

model_path2 = 'model_multi.h5' # load .h5 Model


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#classes2 = {0:"APHIDS",1:"ARMYWORM",2:"BEETLE",3:"BOLLWORM",4:"GRASSHOPPER",5:"MITES",6:"MOSQUITO",7:"SAWFLY",8:"STEM BORER"}
CTS1 = load_model(model_path1, custom_objects={'f1_score' : f1_m, 'precision_score' : precision_m, 'recall_score' : recall_m}, compile=False)

CTS2 = load_model(model_path2, custom_objects={'f1_score' : f1_m, 'precision_score' : precision_m, 'recall_score' : recall_m}, compile=False)



from keras.preprocessing.image import load_img, img_to_array

def model_predict1(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(128,128))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    print(result)
    #prediction = classes2[result]  
    
    if result == 0:
        return "COVID-19 include bilateral peripheral and basal multifocal airspace opacities (ground-glass opacity (GGO) and consolidation).","result.html"        
    elif result == 1:
        return "NON-COVID19","result.html"
    

    
def model_predict2(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(128,128))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    print(result)
    #prediction = classes2[result]  
    
    if result == 0:
        return "COVID-19 include bilateral peripheral and basal multifocal airspace opacities (ground-glass opacity (GGO) and consolidation).","result.html"        
    elif result == 1:
        return "NON-COVID19","result.html"

    elif result == 2:
        return "VIRUS as 4- to 10-mm, poorly defined nodules and patchy areas of peribronchial ground-glass opacity and airspace consolidation, with variable hyperinflation.","result.html"        


@app.route("/about")
def about():
    return render_template("about.html")

   
@app.route("/about1")
def about1():
    return render_template("about1.html")

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/index1')
def index1():
	return render_template('index1.html')

@app.route('/index2')
def index2():
	return render_template('index2.html')

@app.route('/index3')
def index3():
	return render_template('index3.html')

@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    
    print("Entered here")
    file = request.files['file'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    print("@@ Predicting class......")
    pred, output_page = model_predict1(file_path,CTS1)
              
    return render_template(output_page, pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)


@app.route('/predict1',methods=['GET','POST'])
def predict1():
    print("Entered")
    
    print("Entered here")
    file = request.files['file'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    print("@@ Predicting class......")
    pred, output_page = model_predict2(file_path,CTS2)
              
    return render_template(output_page, pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)



@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook1")
def notebook1():
    return render_template("Class_Binary.html")

@app.route("/notebook2")
def notebook2():
    return render_template("Class_Multi.html")

@app.route("/notebook3")
def notebook3():
    return render_template("Detect_Bin.html")

@app.route("/notebook4")
def notebook4():
    return render_template("Detect_multi.html")



model = torch.hub.load("ultralytics/yolov5", "custom", path = "bin_best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45


model1 = torch.hub.load("ultralytics/yolov5", "custom", path = "multi_best.pt", force_reload=True)

model1.eval()
model1.conf = 0.5  
model1.iou = 0.45


from io import BytesIO

@app.route("/predict2", methods=["GET", "POST"])
def predict2():
    """
    The function takes in an image, runs it through the model, and then saves the output image to a
    static folder
    :return: The image is being returned.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=415)
        results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")
    return render_template("index2.html")


@app.route("/predict3", methods=["GET", "POST"])
def predict3():
    """
    The function takes in an image, runs it through the model, and then saves the output image to a
    static folder
    :return: The image is being returned.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model1(img, size=415)
        results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image1.jpg", format="JPEG")
        return redirect("static/image1.jpg")
    return render_template("index3.html")



if __name__ == '__main__':
    app.run(debug=False)