import os
#import urllib.request
#import cv2
#import numpy as np
#from app import app
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename
#import requests
#import json
#from gradcam import *
#import base64

from libauc.models import densenet121 as DenseNet121
import torch
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)


checkpoint = torch.load("./aucm_pretrained_model.pth", map_location=torch.device('cpu'))
model = DenseNet121(pretrained=True, last_activation=False, activations='relu', num_classes=5)
model.load_state_dict(checkpoint)
resp = model.eval()
#print(resp)

allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/radiografia', methods=['POST'])
def upload_file():
	# check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        #obtener nombre del archivo----------------
        filename = secure_filename(file.filename)
        print("Nombre de archivo: ",filename)

        #obtener extension de archivo (.jpg)-------
        extension= os.path.splitext(filename)
        print('extension:', extension)
        ext=str(extension)

        #Img---------------------------------
        # Lee la imagen enviada desde Postman
        file = request.files['file']  
        # Convierte el archivo en una matriz de bytes
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        # Carga la imagen utilizando OpenCV
        imageCSV = cv2.imdecode(file_bytes, cv2.COLOR_GRAY2RGB)

        #imageCSV = cv2.imread(file_bytes, 0)
        print("Imagen cargada: ",imageCSV)
        image = Image.fromarray(imageCSV)

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        img= image.transpose((2, 0, 1)).astype(np.float32)
        input_tensor = torch.from_numpy(img)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        return str(input_tensor)
    else:               
        resp = jsonify({'message' : 'Allowed file types are, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp   



if __name__ == '__main__':
    app.run()