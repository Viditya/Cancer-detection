import sqlite3
import numpy as np
import sys
import os
import glob
import re

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename


app = Flask(__name__)
model_path = 'vgg19.h5'

## Loading the model
model = load_model(model_path, compile=False)
model.make_predict_function()


## Preprocessing function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


# @app.route('/')
# def hello():
#     return 'Hello, World!'

@app.route('/', methods = ['GET'])
def index():
    conn = get_db_connection()
    track = conn.execute('SELECT * FROM tracker').fetchall()
    conn.close()
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        #Get the file from the POST
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        ### Make the predictions
        pred = model_predict(file_path, model)
        pred_class = decode_predictions(pred, top=1)
        result = str('This is a '+ pred_class[0][0][1] + ' with probability ' + str(round(100*pred_class[0][0][2])) + '%')
        print(result)        
        return result
    return None


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/create', methods=('GET', 'POST'))
def create():
    ip = ''
    search_text = ''
    conn = get_db_connection()
    conn.execute('INSERT INTO tracker (ip, search_text) VALUES (?, ?)',
                    (ip , search_text))
    conn.commit()
    conn.close()

    return render_template('create.html')