import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow.keras.utils as keras_utils
import tensorflow.keras as keras

# Define the custom layer if it's not defined in your code
# Example:
# from my_custom_layers import FixedDropout

# Or if you're using a lambda function for FixedDropout, you can redefine it during loading
def fixed_dropout(rate, **kwargs):
    return keras.layers.Dropout(rate, **kwargs)

with keras_utils.custom_object_scope({'FixedDropout': fixed_dropout}):
    model = load_model('Skin__diseases_effnet_model.h5')

# Define the class labels
class_labels = ['acne', 'atopic', 'bcc']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_disease(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    predicted_probability = prediction[0, predicted_class_index]
    return predicted_class, predicted_probability, prediction  # Return prediction probabilities for debugging

# Streamlit app
def main():
    st.title('Skin Disease Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        disease, probability, prediction_probs = predict_disease(img)
        st.write(f"Prediction: {disease} (Probability: {probability:.2f})")
        #st.write("Prediction Probabilities:", prediction_probs)  # Debugging output for prediction probabilities

if __name__ == '__main__':
    main()
