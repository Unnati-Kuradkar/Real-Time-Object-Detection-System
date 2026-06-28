import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano

st.set_page_config(page_title="Object Detection")

st.title("🚗 RF-DETR Object Detection")

@st.cache_resource
def load_model():
    return RFDETRNano()

model = load_model()

input_type = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# ---------------- IMAGE ----------------
if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image")

        detections = model.predict(image_np)

        st.success(
            f"Objects Detected: {len(detections.xyxy)}"
        )

# ---------------- VIDEO ----------------
else:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        st.video(uploaded_video)
        st.success("✅ Video uploaded successfully")
