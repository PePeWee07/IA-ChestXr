import os
from flask import Flask, request, redirect, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert

import torch
from torch import nn
from torchvision import transforms
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
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
CORS(app)

checkpoint = torch.load("./aucm_pretrained_model.pth", map_location=torch.device('cpu'))
model = DenseNet121(pretrained=True, last_activation=False, activations='relu', num_classes=5)
model.load_state_dict(checkpoint)
resp = model.eval()

allowed_extensions = {'dcm'}

# Función para comprobar la extensión del archivo
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Función para convertir un archivo DICOM a JPEG
def dcm_to_jpg(input_path, output_path):
    # Leer el archivo DICOM
    ds = pydicom.dcmread(input_path, force=True)

    # Obtener la matriz de píxeles
    pixel_array = ds.pixel_array

    # Crear una imagen PIL a partir de la matriz de píxeles
    # Convertir la matriz de píxeles a un modo de imagen compatible con JPEG (por ejemplo, "L" para escala de grises de 8 bits)
    image = Image.fromarray(pixel_array).convert("L")

    # Guardar la imagen en formato JPEG
    image.save(output_path)

# Ruta de salida para el archivo JPEG
output_file = "./upload/radiografia.jpg"

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
        # Llamar a la función para convertir el archivo DICOM a JPEG
        dcm_to_jpg(file, output_file)

        
        #file_bytes = np.asarray(bytearray(output_file.read()), dtype=np.uint8)

        # Leer el contenido del archivo en una matriz de bytes
        with open(output_file, "rb") as file:
            file_bytes_array = file.read()

        # Convertir la matriz de bytes en un arreglo de NumPy
        file_bytes= np.frombuffer(file_bytes_array, dtype=np.uint8)

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
                        plt.savefig('./predictions/Cardiomegaly.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        # Cerrar la figura
                        plt.close()
                if(i==1):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Edema", color='white')
                        plt.savefig('./predictions/Edema.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()
                if(i==2):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Consolidation", color='white')
                        plt.savefig('./predictions/Consolidation.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()
                if(i==3):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Atelectasis", color='white')
                        plt.savefig('./predictions/Atelectasis.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()
                if(i==4):
                        plt.imshow(result, interpolation='bicubic')
                        plt.axis('off')
                        plt.title("Pleural Effusion", color='white')
                        plt.savefig('./predictions/Pleural Effusion.jpg', format='jpg', dpi=600, facecolor=fig.get_facecolor())
                        #plt.show()
                        plt.close()                  
        
        #evitar el redondeo de los números
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        
        #return in base64 all the images genarated
        resp = jsonify([
                {
                    'nombre': 'Atelectasis',
                    'imagen': base64.b64encode(open('./predictions/Atelectasis.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][3].item())
                },
                {
                    'nombre': 'Cardiomegaly',
                    'imagen': base64.b64encode(open('./predictions/Cardiomegaly.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][0].item())
                },
                {
                    'nombre': 'Consolidation',
                    'imagen': base64.b64encode(open('./predictions/Consolidation.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][2].item())
                },
                {
                    'nombre': 'Edema',
                    'imagen': base64.b64encode(open('./predictions/Edema.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][1].item())
                },
                {
                    'nombre': 'Pleural Effusion',
                    'imagen': base64.b64encode(open('./predictions/Pleural Effusion.jpg', 'rb').read()).decode('utf-8'),
                    'porcentaje': "{:.4f}".format(out[0][4].item())
                }
            ])

        resp.status_code = 201
        return resp
    else:               
        resp = jsonify({'message' : 'Allowed file types .dcm'})
        resp.status_code = 400
        return resp   


@app.route('/convert', methods=['POST'])
def convert_image():
    try:
        # Verificar si se envió una imagen JPEG
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})
        
        file = request.files['file']
        # Guardar la imagen JPEG en disco
        image_path = './convert-to-dm/img.jpg'
        file.save(image_path)

        # Your input file here
        INPUT_FILE = image_path

        # Name for output DICOM
        dicomized_filename = 'convertImg.dcm'

        # Ruta de la carpeta de salida
        output_folder = 'convert-to-dm'
        os.makedirs(output_folder, exist_ok=True)

        # Ruta completa del archivo DICOM de salida
        output_path = os.path.join(output_folder, dicomized_filename)

        # Load image with Pillow
        img = Image.open(INPUT_FILE)
        width, height = img.size
        print("File format is {} and size: {}, {}".format(img.format, width, height))

        #ds = Dataset()
        ds = pydicom.dcmread('./convert-to-dm/pre-existing.dcm') # pre-existing dicom file

        np_frame = np.array(img.getdata(), dtype=np.uint8)
        np_frame = np_frame.reshape((img.height, img.width))
        
        ds.file_meta = Dataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        ds.file_meta.ImplementationClassUID = "1.2.3.4"

        ds.PatientName = 'Created'

        ds.Rows = img.height
        ds.Columns = img.width
        ds.PhotometricInterpretation = "MONOCHROME1"
        if np_frame.shape[1] == 3:
            ds.SamplesPerPixel = 3
        else:
            ds.SamplesPerPixel = 1
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0
        ds.NumberOfFrames = 1

        ds.PixelData = np_frame.tobytes()
        # Ajustar WindowCenter y WindowWidth
        ds.WindowCenter = -896.5
        ds.WindowWidth = 255

        ds.SOPClassUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()

        ds.PixelData = np_frame

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.save_as(output_path, write_like_original=False)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()