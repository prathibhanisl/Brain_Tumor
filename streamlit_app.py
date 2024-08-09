# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

# Define the ImagePreprocessor class
class ImagePreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._preprocess_image(image) for image in X])

    def _preprocess_image(self, image):
        if image.ndim == 2:  # Grayscale image
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = tf.image.resize(image, self.target_size, preserve_aspect_ratio=True)
        image = tf.image.resize_with_pad(image, self.target_size[0], self.target_size[1])
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image

# Load the preprocessor pipeline and model
@st.cache_resource
def load_pipeline_and_model():
    preprocessor = joblib.load('pipeline.joblib')
    model_cnn = load_model('model.h5')
    return preprocessor, model_cnn

# Function to predict the class of an uploaded image
def predict_image_class(image, preprocessor, model_cnn):
    # Class number to label mapping
    class_labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}

    # Convert the uploaded image to grayscale
    image = image.convert('L')
    image = np.array(image)

    # Preprocess the image
    preprocessed_image = preprocessor.transform([image])

    # Predict the class using the loaded Keras model
    prediction = model_cnn.predict(preprocessed_image)

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map the class number to the label
    predicted_label = class_labels[predicted_class]
    return predicted_label

def main():
    st.title("Brain Tumor Classification Tool")
    st.write("Upload an MRI to determine classification.")
    st.write("Note: The model has an accuracy of 72%.")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an MRI...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image in a smaller size
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

        # Load the pipeline and model
        preprocessor, model_cnn = load_pipeline_and_model()

        # Make prediction
        predicted_label = predict_image_class(image, preprocessor, model_cnn)
        st.write(f"The image above indicates that the patient has :{predicted_label}.")

if __name__ == "__main__":
    main()
