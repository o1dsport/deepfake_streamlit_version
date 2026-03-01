import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from face_utils import detect_face
from gradcam_utils import make_gradcam_heatmap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Deepfake Detector")

st.title("AI Powered Deepfake Detection System")

model = tf.keras.models.load_model("deepfake_model.h5")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Original Image")

    face = detect_face(image_np)

    face_resized = cv2.resize(face, (224,224))
    face_array = np.array(face_resized) / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    prediction = model.predict(face_array)[0][0]

    confidence = prediction * 100

    if confidence > 70:
        risk = "HIGH RISK"
    elif confidence > 50:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    if prediction > 0.5:
        st.error(f"Deepfake Detected ({confidence:.2f}%)")
    else:
        st.success(f"Real Image ({100-confidence:.2f}%)")

    st.write(f"Risk Level: {risk}")

    heatmap = make_gradcam_heatmap(face_array, model)
    heatmap = cv2.resize(heatmap, (224,224))

    plt.imshow(face_resized)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')

    st.pyplot(plt)
