import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv
import cv2
import tempfile

st.set_page_config(
    page_title="RF-DETR Object Detection",
    layout="wide"
)

st.title("RF-DETR Object Detection")

# Load model only once
@st.cache_resource
def load_model():
    model = RFDETRBase()
    return model

model = load_model()

option = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# ==========================
# IMAGE DETECTION
# ==========================
if option == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting objects..."):

            detections = model.predict(image_np)

            box_annotator = sv.BoxAnnotator()

            annotated_image = box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )

        st.image(
            annotated_image,
            caption="Detected Objects",
            use_container_width=True
        )

        st.subheader("Detected Objects")

        for name, conf in zip(
            detections.data["class_name"],
            detections.confidence
        ):
            st.write(f"✅ {name} ({conf:.2f})")

# ==========================
# VIDEO DETECTION
# ==========================
if option == "Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.info("Processing video... Please wait.")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)

        stframe = st.empty()

        box_annotator = sv.BoxAnnotator()

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process only every 10th frame
            if frame_count % 10 != 0:
                continue

            # Resize frame for faster inference
            frame = cv2.resize(frame, (640, 480))

            detections = model.predict(frame)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )

            stframe.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        st.success("Video processing completed!")
