from flask import Flask, request, render_template
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.models import load_model
import segmentation_models as sm
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your model
weights = [0.1666] * 6  # Adjust if necessary
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

model_path = r"C:\Users\sriva\Videos\dubai segmentation\satellite_standard_unet_100epochs.hdf5"
custom_objects = {
    "dice_loss_plus_1focal_loss": total_loss,
    "jacard_coef": jacard_coef
}
model = load_model(model_path, custom_objects=custom_objects)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400
    
    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess the image
    test_img = Image.open(file_path)
    desired_width = 256
    desired_height = 256
    test_img = test_img.resize((desired_width, desired_height))
    test_img = np.array(test_img)
    test_img_input = np.expand_dims(test_img, 0)

    # Make the prediction
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # Save the predicted image
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + file.filename), predicted_img, cmap='jet')

    return render_template('index.html', uploaded_image=file.filename, predicted_image='predicted_' + file.filename)

if __name__ == '__main__':
    app.run(debug=True)
