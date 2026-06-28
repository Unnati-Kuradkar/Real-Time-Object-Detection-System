import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image
import supervision as sv
import torch

from rfdetr import RFDETRNano

st.set_page_config(
    page_title="RF-DETR Object Detection",
    layout="wide"
)

st.title("🚗 RF-DETR Object Detection")

# ----------------------------
# Load Model Only Once
# ----------------------------
@st.cache_resource
def load_model():
    model = RFDETRNano()

    try:
        model.optimize_for_inference(
            dtype=torch.float16
        )
    except:
        pass

    return model


model = load_model()

# ----------------------------
# Mode Selection
# ----------------------------
option = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# =====================================================
# IMAGE DETECTION
# =====================================================
if option == "Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        with st.spinner("Detecting Objects..."):

            detections = model.predict(image_np)

            box_annotator = sv.BoxAnnotator()

            annotated_image = box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )

        st.image(
            annotated_image,
            caption="Detected Objects",
            width="stretch"
        )

        st.subheader("Detected Objects")

        try:
            for name, conf in zip(
                detections.data["class_name"],
                detections.confidence
            ):
                st.write(
                    f"✅ {name} ({conf:.2f})"
                )
        except:
            st.info("Objects detected")

# =====================================================
# VIDEO DETECTION
# =====================================================
else:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button("Start Detection"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tmp_file:

                tmp_file.write(
                    uploaded_video.read()
                )

                video_path = tmp_file.name

            cap = cv2.VideoCapture(video_path)

            frame_placeholder = st.empty()

            frame_count = 0

            # Process every 60th frame
            frame_skip = 60

            st.info(
                "⚡ Fast Mode Enabled"
            )

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if frame_count % frame_skip != 0:
                    continue

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                detections = model.predict(
                    frame_rgb
                )

                box_annotator = sv.BoxAnnotator()

                annotated_frame = (
                    box_annotator.annotate(
                        scene=frame_rgb.copy(),
                        detections=detections
                    )
                )

                frame_placeholder.image(
                    annotated_frame,
                    width="stretch"
                )

            cap.release()

            st.success(
                "✅ Detection Complete"
            )
