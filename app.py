import io
import numpy as np
from PIL import Image
import streamlit as st
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
  input_image = Image.open(uploaded_file).convert("RGB")

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Original Image")
    st.image(input_image, use_container_width=True)

  with st.spinner("Analyzing foliar tissue..."):
    results = model.predict(
        source=input_image,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=1024,
        retina_masks=True,
    )[0]

    res_plotted = results.plot(boxes=True, masks=True, conf=True)
    res_image = Image.fromarray(res_plotted)

  with col2:
    st.subheader("Segmented Pathology")
    st.image(res_image, use_container_width=True)

  st.divider()
  st.subheader("Diagnostic Summary")

  boxes = results.boxes
  masks = results.masks

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

    cols = st.columns(len(unique_classes) + 1)
    cols[0].metric("Total Lesions", len(detected_classes))
    for idx, (cls_name, count) in enumerate(zip(unique_classes, counts)):
      cols[idx + 1].metric(
          cls_name.replace("_", " ").title(), f"{count} spots"
      )

    if masks is not None:
      total_pixels = input_image.size[0] * input_image.size[1]
      combined_mask = np.zeros(
          (input_image.size[1], input_image.size[0]), dtype=bool
      )
      for m in masks.data.cpu().numpy():
        m_resized = (
            np.array(
                Image.fromarray(m.astype(np.uint8)).resize(
                    input_image.size, resample=Image.NEAREST
                )
            )
            > 0
        )
        combined_mask = np.logical_or(combined_mask, m_resized)
      severity_pct = (np.sum(combined_mask) / total_pixels) * 100.0
      st.markdown(
          f"**Estimated Leaf Surface Infection Severity:** `{severity_pct:.2f}%`"
      )

    breakdown_data = [
        {
            "Lesion ID": f"#{i+1}",
            "Pathology": names[cls_ids[i]].replace("_", " ").title(),
            "Confidence": f"{confidences[i]*100:.2f}%",
        }
        for i in range(len(boxes))
    ]
    st.dataframe(breakdown_data, use_container_width=True)

    buf = io.BytesIO()
    res_image.save(buf, format="PNG")
    st.download_button(
        "📥 Download Segmentation Result",
        buf.getvalue(),
        "tomato_detection.png",
        "image/png",
    )
