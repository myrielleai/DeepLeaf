import os
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DeepLeaf | Tomato Disease Segmentation",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Light Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #0f2027 100%);
        padding: 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
    }

    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 1.5rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .card {
            background: #1e293b;
            border: 1px solid #334155;
            color: #f8fafc;
        }
    }

    .disease-title {
        color: #ef4444;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .healthy-title {
        color: #10b981;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .stat-box {
        text-align: center;
        padding: 1rem;
        background: rgba(99, 102, 241, 0.08);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #6366f1;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Disease Knowledge Base with Management Advice
DISEASE_INFO = {
    "Bacterial Spot": {
        "symptoms": "Small, dark, water-soaked spots on leaves that turn brown/black with yellow halos.",
        "organic": "Apply copper-based fungicides or neem oil. Remove affected leaves.",
        "prevention": "Avoid overhead watering, practice 2-3 year crop rotation, use disease-free seeds."
    },
    "Early Blight": {
        "symptoms": "Concentric rings (target pattern) on older leaves with yellowing foliage.",
        "organic": "Copper fungicides, bio-fungicides (Bacillus subtilis), prune bottom leaves.",
        "prevention": "Mulch soil surface to prevent spore splashback, maintain adequate plant spacing."
    },
    "Late Blight": {
        "symptoms": "Large, dark green to brown water-soaked lesions with white mold on underside in high humidity.",
        "organic": "Copper spray immediately at first sign; remove and destroy severely infected plants.",
        "prevention": "Plant resistant varieties, ensure high airflow, avoid wet foliage."
    },
    "Leaf Mold": {
        "symptoms": "Pale green or yellow spots on leaf tops with olive-green velvet mold underneath.",
        "organic": "Bio-fungicides, copper sprays; improve ventilation in greenhouse settings.",
        "prevention": "Maintain humidity below 85%, increase ventilation and spacing."
    },
    "Septoria Leaf Spot": {
        "symptoms": "Numerous small circular spots with gray centers and dark brown borders.",
        "organic": "Apply copper or sulfur fungicides; trim lower infected leaves.",
        "prevention": "Keep foliage dry, control weeds, sanitize garden tools after use."
    },
    "Spider Mites": {
        "symptoms": "Yellow speckled leaves, fine webbing under leaves, leaves bronzing or dying.",
        "organic": "Insecticidal soap, neem oil, predatory mites (Phytoseiulus persimilis).",
        "prevention": "Keep plants well-watered; spray leaf undersides with strong water jets."
    },
    "Target Spot": {
        "symptoms": "Brown circular spots with light brown centers and dark margins on foliage.",
        "organic": "Copper-based fungicides or bio-pesticides.",
        "prevention": "Prune foliage for ventilation, avoid field flooding."
    },
    "Yellow Leaf Curl Virus": {
        "symptoms": "Upward curling leaves, yellow leaf margins, stunted growth, flower drop.",
        "organic": "Control whitefly vectors using yellow sticky traps, neem oil, insect netting.",
        "prevention": "Use virus-resistant cultivars, destroy infected plants immediately."
    },
    "Mosaic Virus": {
        "symptoms": "Mottled light/dark green mosaic patterns, fern-like leaf distortion.",
        "organic": "No cure once infected. Remove plant to protect surrounding crop.",
        "prevention": "Wash hands with milk/soap before handling plants, control aphids."
    },
    "Healthy": {
        "symptoms": "Vibrant green leaves, crisp leaf margins, uniform foliage without discoloration.",
        "organic": "Continue balanced organic fertilization (N-P-K) and compost tea.",
        "prevention": "Maintain regular watering schedule and optimal soil nutrients."
    }
}

MODEL_PATH = "best.pt"

@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO segmentation model if present."""
    if not os.path.exists(model_path):
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header Section
    st.markdown("""
    <div class="main-header">
        <h1>🍅 DeepLeaf: Tomato Disease Segmentation</h1>
        <p>AI-Powered Instance Segmentation & Diagnostic System for Tomato Crop Health</p>
        <span class="badge">Connected Repository: DeepLeaf</span>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    model = load_yolo_model(MODEL_PATH)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        if model is None:
            st.warning(f"⚠️ Model file `{MODEL_PATH}` not detected in root directory.")
            st.info("💡 Please place your trained `best.pt` file inside the project folder to enable live AI detection.")
        else:
            st.success("✅ `best.pt` model loaded successfully!")

        st.subheader("Inference Settings")
        conf_thresh = st.slider("Confidence Threshold", min_value=0.05, max_value=1.00, value=0.25, step=0.05)
        iou_thresh = st.slider("IoU NMS Threshold", min_value=0.05, max_value=1.00, value=0.45, step=0.05)
        
        st.subheader("Visualization Options")
        show_masks = st.checkbox("Show Instance Masks", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_labels = st.checkbox("Show Labels & Confidence", value=True)

        st.divider()
        st.markdown("### 📌 About DeepLeaf")
        st.markdown(
            "DeepLeaf utilizes state-of-the-art **YOLO instance segmentation** "
            "to pinpoint, delineate, and diagnose leaf diseases in tomato plants."
        )

    # Input Section
    st.subheader("📥 Input Image")
    input_type = st.radio("Select Input Source:", ["Upload Image", "Use Camera Capture"], horizontal=True)

    image = None
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Take a photo of the tomato leaf")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")

    if image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### 🔍 Segmentation & Analysis")

            if model is None:
                st.error("Cannot perform inference because `best.pt` is missing.")
                st.info("Add your `best.pt` file into the current directory to enable live predictions.")
            else:
                with st.spinner("Processing image with DeepLeaf YOLO segmentation..."):
                    start_time = time.time()
                    
                    # Convert PIL image to numpy array
                    img_array = np.array(image)

                    # Run Ultralytics YOLO Prediction
                    results = model.predict(
                        source=img_array,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False
                    )
                    
                    inference_time = (time.time() - start_time) * 1000
                    result = results[0]

                    # Generate plot visualization
                    res_plotted = result.plot(
                        conf=show_labels,
                        boxes=show_boxes,
                        masks=show_masks
                    )
                    
                    # Convert BGR back to RGB for Streamlit
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, use_container_width=True)
                    st.caption(f"⚡ Inference completed in {inference_time:.1f} ms")

        # Diagnostics & Detailed Breakdown
        if model is not None and len(results) > 0:
            st.divider()
            st.subheader("📊 Diagnostic Summary")

            boxes = result.boxes
            masks = result.masks

            num_detections = len(boxes) if boxes is not None else 0

            # Metrics Summary Row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{num_detections}</div>
                    <div class="stat-label">Detections Found</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                avg_conf = f"{float(boxes.conf.mean()) * 100:.1f}%" if num_detections > 0 else "N/A"
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{avg_conf}</div>
                    <div class="stat-label">Average Confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col3:
                mask_coverage = "N/A"
                if masks is not None and masks.data is not None:
                    # Calculate percentage of image area covered by masks
                    combined_mask = np.any(masks.data.cpu().numpy(), axis=0)
                    coverage_pct = (np.sum(combined_mask) / combined_mask.size) * 100
                    mask_coverage = f"{coverage_pct:.1f}%"
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{mask_coverage}</div>
                    <div class="stat-label">Infected Leaf Coverage</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            if num_detections > 0:
                # Group detected classes
                names = result.names
                detected_classes = [names[int(cls)] for cls in boxes.cls.cpu().numpy()]
                unique_classes = set(detected_classes)

                st.markdown("### 🌿 Detected Disease Insights & Treatment Guidelines")

                for disease_name in unique_classes:
                    # Look up disease info or default
                    matched_key = None
                    for key in DISEASE_INFO:
                        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
                            matched_key = key
                            break
                    
                    info = DISEASE_INFO.get(matched_key, {
                        "symptoms": "Identified by DeepLeaf YOLO model.",
                        "organic": "Consult local agricultural extension office for tailored organic solutions.",
                        "prevention": "Maintain proper plant sanitation, crop rotation, and moisture management."
                    })

                    is_healthy = "healthy" in disease_name.lower()
                    card_title_class = "healthy-title" if is_healthy else "disease-title"
                    icon = "✅" if is_healthy else "⚠️"

                    st.markdown(f"""
                    <div class="card">
                        <div class="{card_title_class}">{icon} {disease_name}</div>
                        <p><strong>Symptoms:</strong> {info['symptoms']}</p>
                        <p><strong>Recommended Treatment:</strong> {info['organic']}</p>
                        <p><strong>Prevention Tip:</strong> {info['prevention']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info("No disease or leaf instances detected above the current confidence threshold. Try adjusting the slider in the sidebar.")

if __name__ == "__main__":
    main()
