import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm
from tensorflow.keras import backend as K

# Initialize the Flask app
app = Flask(__name__)

# Load the model
weights = [0.1666] * 6
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

model_path = "modal/satellite_standard_unet_100epochs.hdf5"
custom_objects = {
    "dice_loss_plus_1focal_loss": total_loss,
    "jacard_coef": jacard_coef
}
model = load_model(model_path, custom_objects=custom_objects)

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Load and process the image
            test_img = Image.open(file)
            desired_width, desired_height = 256, 256
            test_img = test_img.resize((desired_width, desired_height))
            test_img = np.array(test_img)

            # Prepare the image for the model
            test_img_input = np.expand_dims(test_img, axis=0)

            # Make the prediction
            prediction = model.predict(test_img_input)
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]

            # Save the predicted image
            plt.imshow(predicted_img)
            plt.axis('off')
            plt.savefig("static/predicted_image.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            return render_template("index.html", predicted_image="static/predicted_image.png")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
