# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
import time
import io # <--- Ensure this is imported

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload directory '{UPLOAD_FOLDER}' ensured to exist.")

model = None
MODEL_PATH = 'model/flower_model.h5'
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please ensure the model file exists and is not corrupted.")

class_names = []
CLASS_NAMES_PATH = 'model/class_names.json'
try:
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Class names loaded successfully from {CLASS_NAMES_PATH}")
    else:
        print(f"Error: Class names file not found at {CLASS_NAMES_PATH}. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading class names from {CLASS_NAMES_PATH}: {e}")
    print("Please ensure the class_names.json file exists and is valid JSON.")

IMG_HEIGHT = 224
IMG_WIDTH = 224

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not class_names:
        print("Prediction request received, but model or class names are not loaded.")
        return jsonify({'error': 'AI model or class names not loaded. Please check backend setup and logs.'}), 500

    if 'file' not in request.files:
        print("Prediction request received, but no 'file' part in the request.")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        print("Prediction request received, but no selected file.")
        return jsonify({'error': 'No selected file'}), 400

    filepath = None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # --- CRITICAL CHANGE: Read file contents into memory stream first ---
            image_bytes = file.read() # Reads the incoming file stream into memory
            image_stream = io.BytesIO(image_bytes) # Creates an in-memory file-like object
            print("Image data read into in-memory stream.")

            # --- Save the image to disk (optional, but good for cleanup) ---
            # Use 'image_bytes' here as 'file.read()' already consumed the stream
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            print(f"File saved temporarily to {filepath}")

            # --- Load and preprocess the image from the in-memory stream ---
            # This is the crucial part that avoids the WinError 32
            img = image.load_img(image_stream, target_size=(IMG_HEIGHT, IMG_WIDTH))

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_index]

            confidence = float(predictions[0][predicted_class_index]) * 100

            # Clean up the uploaded file after prediction
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Temporary file {filepath} removed.")
                except OSError as ose:
                    print(f"Failed to remove temporary file {filepath} during cleanup: {ose}")

            all_predictions = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}

            print(f"Prediction successful: {predicted_class_name} with {confidence:.2f}% confidence.")
            return jsonify({
                'prediction': predicted_class_name,
                'confidence': confidence,
                'all_predictions': all_predictions
            })

        except Exception as e:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Temporary file {filepath} removed due to error.")
                except OSError as ose:
                    print(f"Failed to remove temporary file {filepath} after error: {ose}")
            print(f"Error during prediction for {filename}: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}. Check server logs for details.'}), 500
    else:
        print(f"Prediction request received for disallowed file type: {file.filename}")
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg, gif'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)
    print("Flask app started.")