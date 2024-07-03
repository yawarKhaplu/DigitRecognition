# app.py
from flask import Flask, render_template, request
import tensorflow as tf
import os
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

model = joblib.load('svm_model.joblib')
model_cnn = tf.keras.models.load_model('hdr.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    model_type = request.form['model']

    # Save the uploaded image
    image_path = file.filename
    file.save(image_path)

    # Process the image for prediction
    processed_img = preprocess_image(image_path,model_type)

    # Implement your model prediction logic based on `model_type`
    prediction_result = predict_image(processed_img, model_type)

    # Delete the saved image after processing
    os.remove(image_path)

    return render_template('result.html', image_path=image_path, prediction=prediction_result)

def preprocess_image(image_path,model):
    # Example: Resize image and convert to numerical array
    if model == 'svm':
        img = Image.open(image_path)
        img = img.resize((28, 28)).convert('L')  # Resize as required by your SVM model
        img_array = np.array(img)   # Convert image to numpy array
        processed_img = img_array.flatten().reshape(1, -1)  # Flatten and reshape for SVM prediction
        return processed_img
    elif model == 'cnn':
        img = Image.open(image_path)
        img = img.resize((28, 28)).convert('L')  # Resize as required by your SVM model
        img_array = np.array(img) / 255.0  # Convert image to numpy array
        processed_img = img_array.flatten().reshape(1, 28, 28, 1)  # Flatten and reshape for SVM prediction
        # processed_img = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return processed_img
def predict_image(processed_img, model_type):
    if model_type == 'svm':
        prediction = model.predict(processed_img)
        return f"SVM Model Prediction: {prediction[0]}"
    elif model_type == 'cnn':
        prediction = model_cnn.predict(processed_img)
        predicted_class = np.argmax(prediction, axis=1)
        return f"{predicted_class[0]}"
    else:
        return "Model not supported"

if __name__ == '__main__':
    app.run(debug=True)
