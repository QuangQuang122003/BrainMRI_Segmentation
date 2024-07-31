import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the custom loss function and dice coefficient
smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Load models
@st.cache_resource
def load_models():
    custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef}
    models = {
        'UNet++': load_model('/content/drive/MyDrive/Colab Notebooks/BrainMRI/Unet_improve/model.h5', custom_objects=custom_objects)
    }
    return models

# Predict function
def predict(model, image):
    image = cv2.resize(image, (256, 256))  # Assuming model input size is 256x256
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.squeeze(prediction)

# Streamlit app
st.title('Brain MRI Tumor Segmentation')

st.sidebar.title('Model Selection')
models = load_models()
model_choice = st.sidebar.selectbox('Choose Model', list(models.keys()))

uploaded_files = st.file_uploader('Upload MRI images', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files is not None and st.button('Segment Tumors'):
    for uploaded_file in uploaded_files:
        image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
        
        model = models[model_choice]
        prediction = predict(model, image)

        st.subheader(f'Segmentation Result for {uploaded_file.name}')
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Original Image', use_column_width=True)
        with col2:
            st.image(prediction, caption='Segmented Tumor', use_column_width=True)
