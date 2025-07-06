# UASvisikomputer

import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load model hanya sekali
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Ganti dengan 'best.pt' jika kamu punya model sendiri

model = load_model()

st.title("🔴 Deteksi Objek Real-Time di Streamlit")
st.markdown("Gunakan kamera untuk mendeteksi objek secara langsung.")

run = st.checkbox('Mulai Deteksi Kamera')

# Buat placeholder untuk hasil video
frame_placeholder = st.empty()

# Mulai webcam jika checkbox diaktifkan
if run:
    cap = cv2.VideoCapture(0)  # 0 = kamera default laptop/PC

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal mengakses kamera.")
            break

        # Deteksi objek
        results = model.predict(frame, conf=0.4)
        frame = results[0].plot()

        # Tampilkan ke Streamlit
        frame_placeholder.image(frame, channels="BGR")

    cap.release()
else:
    st.info("Aktifkan checkbox di atas untuk memulai kamera.")
