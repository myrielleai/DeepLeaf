import io
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Tomato Foliar Disease Instance Segmentation",
    page_icon="🍅",
    layout="wide",
)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


st.title("Tomato Foliar Disease Detection via YOLO26 Instance Segmentation")
st.markdown("High-resolution diagnostic engine for tomato leaf pathologies.")

# Sidebar settings
st.sidebar.header("⚙️ Model Controls")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.65,
    step=0.05,
    help=(
        "Set to 0.65 for conservative filtering, or 0.25 (F1-optimal) for"
        " micro-lesion sensitivity."
    ),
)
iou_thresh = st.sidebar.slider(
    "IoU Threshold", min_value=0.10, max_value=0.90, value=0.60, step=0.05
)

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model weights: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a Tomato Leaf Image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Free cached GPU/CPU tensors from prior predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    input_image = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = input_image.size

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(input_image, width="stretch")

    with st.spinner("Analyzing foliar tissue..."):
        try:
            results = model.predict(
                source=input_image,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=1024,
                retina_masks=True,
                verbose=False,
            )[0]

            # results.plot() returns BGR; convert to RGB
            res_plotted_bgr = results.plot(boxes=True, masks=True, conf=True)
            res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
            res_image = Image.fromarray(res_plotted_rgb)

        except Exception as err:
            st.error(f"Inference error: {err}")
            st.stop()

    with col2:
        st.subheader("Segmented Pathology")
        st.image(res_image, width="stretch")

    st.divider()
    st.subheader("Diagnostic Summary")

    boxes = results.boxes
    masks = results.masks

    # Guard: No detections found
    if boxes is None or len(boxes) == 0:
        st.info(
            f"No lesions detected at confidence {conf_thresh:.2f}. "
            "Try lowering the slider toward 0.25 for micro-lesion sensitivity."
        )
    else:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        names = results.names

        detected_classes = [names[c] for c in cls_ids]
        unique_classes, counts = np.unique(detected_classes, return_counts=True)

        # Safe metric rendering (max 4 columns per row to avoid visual breaking)
        st.markdown(f"**Total Lesions Detected:** `{len(detected_classes)}`")
        metric_cols = st.columns(min(len(unique_classes), 4))
        for idx, (cls_name, count) in enumerate(zip(unique_classes, counts)):
            metric_cols[idx % 4].metric(
                label=cls_name.replace("_", " ").title(),
                value=f"{count} lesions",
            )

        # High-performance vectorized mask merging (prevents memory crash)
        if masks is not None and len(masks.data) > 0:
            try:
                # Merge all binary masks along axis 0 in PyTorch/NumPy directly
                raw_masks = masks.data.cpu().numpy()  # shape: (N, H_mask, W_mask)
                merged_small = np.any(raw_masks > 0.5, axis=0).astype(np.uint8)

                # Resize once instead of resizing N separate times
                merged_full = cv2.resize(
                    merged_small,
                    (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                affected_pixels = int(np.count_nonzero(merged_full))
                total_pixels = img_w * img_h
                severity_pct = (affected_pixels / total_pixels) * 100.0

                st.metric("Total Surface Lesion Area", f"{severity_pct:.2f}%")
            except Exception as e:
                st.warning(f"Could not compute lesion surface area: {e}")

        # Lesion tabular breakdown
        breakdown_data = [
            {
                "Lesion ID": f"#{i+1}",
                "Pathology": names[cls_ids[i]].replace("_", " ").title(),
                "Confidence": f"{confidences[i]*100:.2f}%",
            }
            for i in range(len(boxes))
        ]
        st.dataframe(breakdown_data, width="stretch")

        buf = io.BytesIO()
        res_image.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Segmentation Result",
            data=buf.getvalue(),
            file_name="tomato_detection.png",
            mime="image/png",
        )
