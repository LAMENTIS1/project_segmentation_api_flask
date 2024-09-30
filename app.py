from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Constants
MODEL_PATH = 'modal/satellite_standard_unet_100epochs.hdf5'
IMG_HEIGHT = 256  # Set according to your model's expected input size
IMG_WIDTH = 256   # Set according to your model's expected input size
NUM_CLASSES = 6   # Set to the actual number of classes in your segmentation model

app = Flask(__name__)

# Load the model at startup
model = tf.keras.models.load_model(MODEL_PATH)

# Define a color map for visualization
def get_color_map(num_classes):
    color_map = plt.get_cmap("hsv", num_classes)
    return color_map

# Load and preprocess the image
def load_and_preprocess_image(img):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if required
    return img_array

# Make predictions
def make_prediction(img):
    processed_img = load_and_preprocess_image(img)
    predictions = model.predict(processed_img)

    # Assuming multi-class segmentation, get the class with the highest score
    predicted_mask = np.argmax(predictions[0], axis=-1).astype(np.uint8)
    return predicted_mask

# Color the predicted mask
def colorize_mask(mask, num_classes):
    color_map = get_color_map(num_classes)
    color_mask = color_map(mask)
    return (color_mask[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8

# Convert the mask to an image that can be sent in response
def mask_to_image(mask):
    img = image.array_to_img(mask)
    return img

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Load the image
        img = image.load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))

        # Make prediction
        predicted_mask = make_prediction(img)

        # Colorize the predicted mask
        colored_mask = colorize_mask(predicted_mask, NUM_CLASSES)

        # Convert mask to image and send as a response
        mask_img = mask_to_image(colored_mask)

        # Save to a BytesIO object and return it as a file
        img_io = BytesIO()
        mask_img.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
