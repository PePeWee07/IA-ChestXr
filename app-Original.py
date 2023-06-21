import os
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename

from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from torchvision import models
import base64

from torchcam.methods import GradCAM,LayerCAM,XGradCAM

from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib
import matplotlib.pyplot as plt
#Esto configurará Matplotlib para que use el backend "Agg", que no requiere una GUI y es adecuado para trabajar en entornos sin interfaz gráfica.
matplotlib.use('Agg')

app = Flask(__name__)


checkpoint = torch.load("./aucm_pretrained_model.pth", map_location=torch.device('cpu'))
model = DenseNet121(pretrained=True, last_activation=False, activations='relu', num_classes=5)
model.load_state_dict(checkpoint)
resp = model.eval()

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

    # Comprobar si se seleccionó un archivo
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

        #from torchcam.methods import GradCAM,LayerCAM,XGradCAM
        cam_extractor = GradCAM(model, 'features.norm5')

        # Lee la imagen enviada desde Postman
        file = request.files['file']  
        # Convierte el archivo en una matriz de bytes
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        # Carga la imagen utilizando OpenCV
        imageCSV = cv2.imdecode(file_bytes, cv2.COLOR_GRAY2RGB)

        #imageCSV = cv2.imread(file_bytes, 0)
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

        #Caraga del modelo----------------------
        out=model(input_tensor)
        print(out)

        #0:Cardiomegaly, 1:Edema, 2:Consolidation, 3:Atelectasis, 4:Pleural Effusion
        for i in range(5):
            out=model(input_tensor)
            cams=cam_extractor(i,out)
            for name, cam in zip(cam_extractor.target_names, cams):
                cam_cpu = cam.squeeze(0).cpu()  # Move to CPU before converting to PIL image
                result = overlay_mask(Image.fromarray(imageCSV).convert("RGB"), to_pil_image(cam_cpu, mode='F'), alpha=0.5)
                #result = overlay_mask(Image.fromarray(image).convert("RGB"), to_pil_image(cam_cpu, mode='F'), alpha=0.5)
                # Crear una figura con fondo negro
                fig = plt.figure(facecolor='black')
                if(i==0):
                        # Mostrar la imagen
                        plt.imshow(result, interpolation='bicubic')
                        # Desactivar los ejes
                        plt.axis('off')
                        # Establecer el título
                        plt.title("Cardiomegaly", color='white')
                        # Guardar la figura con fondo negro
                        plt.savefig('Cardiomegaly.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        # Cerrar la figura
                        plt.close()
                if(i==1):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Edema", color='white')
                        plt.savefig('Edema.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()
                if(i==2):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Consolidation", color='white')
                        plt.savefig('Consolidation.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()
                if(i==3):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Atelectasis", color='white')
                        plt.savefig('Atelectasis.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()
                if(i==4):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Pleural Effusion", color='white')
                        plt.savefig('Pleural Effusion.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()                  
        
        #evitar el redondeo de los números
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        
        #return in base64 all the images genarated
        resp = jsonify([
                {
                    'nombre': 'Atelectasis',
                    'imagen': base64.b64encode(open('./Atelectasis.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][3].item())
                },
                {
                    'nombre': 'Cardiomegaly',
                    'imagen': base64.b64encode(open('./Cardiomegaly.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][0].item())
                },
                {
                    'nombre': 'Consolidation',
                    'imagen': base64.b64encode(open('./Consolidation.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][2].item())
                },
                {
                    'nombre': 'Edema',
                    'imagen': base64.b64encode(open('./Edema.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][1].item())
                },
                {
                    'nombre': 'Pleural Effusion',
                    'imagen': base64.b64encode(open('./Pleural Effusion.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][4].item())
                }
            ])

        resp.status_code = 201
        return resp
    else:               
        resp = jsonify({'message' : 'Allowed file types are, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp   



if __name__ == '__main__':
    app.run()