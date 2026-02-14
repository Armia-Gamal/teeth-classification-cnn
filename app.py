import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🦷 Oral Disease AI")
st.sidebar.image("images/output.png", use_container_width=True)

st.sidebar.markdown("""
### About This App
This AI model detects oral diseases from medical images.

### Model Details
- CNN-based Deep Learning Model
- Input Size: 256 × 256
- Number of Classes: 7

### Detected Conditions
- CaS  
- CoS  
- Gum  
- MC  
- OC  
- OLP  
- OT  

---
Developed by **Armia Gamal** ❤️
""")

# ===============================
# Load Model (Important)
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_model.h5")

model = load_model()

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title("Image Classification Test")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
