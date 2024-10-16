import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
import tensorflow as tf
import base64
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
from pathlib import Path
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model
# Load the trained SVM model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Load the model
loaded_model = load_model('lstm_autoencoder_model.h5')


 

app = Flask(__name__)
app.secret_key = os.urandom(24)
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/")
def home():
    return render_template("main.html")
@app.route("/bhome")
def bhome():
    return render_template("bhome.html")
@app.route("/bl")
def bl():
    return render_template("blogin.html")
@app.route("/br")
def br():
    return render_template("breg.html")
@app.route("/log")
def ll():
    return render_template("main.html")
@app.route("/p")
def p():
    return render_template("p.html")
@app.route("/bregister",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        add=request.form['Location']
        ph=request.form['Phone no']
        status = Buyer_reg(username,email,password,add,ph) 
        if status == 1:
            return render_template("blogin.html")
        else:
            return render_template("breg.html",m1="failed")        
    

@app.route("/blogin",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = Buyer_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1: 
            session['username'] = request.form['username']                                     
            return render_template("bhome.html", m1="sucess")
        else:
            return render_template("blogin.html", m1="Login Failed")

# Load the trained model
loaded_model = load_model('lstm_autoencoder_model.h5')

@app.route("/pre", methods=['POST', 'GET'])
def pre():
    if request.method == 'POST':
        features = request.form['inputData']
        
        # Convert the input string into a list of floats
        features_list = [float(x) for x in features.split()]
        
        # Reshape it to match the model's expected input format
        custom_input = np.array([features_list])
        
        # Load the dataset to fit the scaler (ensure the data has been loaded correctly)
        data = pd.read_csv('MIT-BIH Arrhythmia Database.csv')
        labels = data['type'].values
        
        # Normalize the input
        scaler = MinMaxScaler()
        scaler.fit(data.drop(['record', 'type'], axis=1))  # Use dataset to fit the scaler
        custom_input_scaled = scaler.transform(custom_input)
        
        # Reshape for LSTM input (1 sample, 1 time step, number of features)
        custom_input_reshaped = custom_input_scaled.reshape(1, 1, -1)
        
        # Make a prediction
        prediction = loaded_model.predict(custom_input_reshaped)
        
        # Encode the labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Get the predicted label
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        
        # Map prediction to a human-readable label
        result = ''
        if predicted_label[0] == "N":
            result = "ECG ARRHYTHMIA IS NOT DETECTED"
        elif predicted_label[0] == "VEB":
            result = "Ventricular Extrasystoles Detected"
        elif predicted_label[0] == "SVEB":
            result = "Supraventricular Extrasystole Detected"
        else:
            result = "Arrhythmia Detected: " + predicted_label[0]
        
        return render_template("result.html", text=result)
    return render_template("bb.html")



if __name__ == "__main__":
    app.run(debug=True)

     
     