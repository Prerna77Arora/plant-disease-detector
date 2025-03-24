import os
import json
import io
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import streamlit as st

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Correct file paths using os.path.join
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices (1).json")


# Load the trained model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)


model = load_model()

# Load class indices (mapping of labels)
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)


# Function to preprocess image before feeding to the model
def load_and_preprocess_image(image, target_size=(224, 224)):
    """Loads an image, resizes it, and converts it to a NumPy array."""
    try:
        # Convert uploaded file to PIL image
        image.seek(0)  # Reset file pointer
        img = Image.open(image)  # Open image
        img = img.convert("RGB")  # Ensure it's in RGB format (handles grayscale issues)
        img = img.resize(target_size)  # Resize image
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except UnidentifiedImageError:
        st.error("❌ Error: Uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")
        return None


# Function to predict image class
def predict_image_class(model, uploaded_file, class_indices):
    """Processes an image and predicts its class using the trained model."""
    image = load_and_preprocess_image(uploaded_file)  # Preprocess image
    if image is None:
        return None, None  # Return None if the image is invalid

    predictions = model.predict(image)  # Get model predictions
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get highest probability class
    predicted_class_name = class_indices[str(predicted_class_index)]  # Convert index to label
    confidence = np.max(predictions) * 100  # Confidence score
    return predicted_class_name, confidence


# Streamlit UI
st.title('🌱 Plant Disease Classifier')

uploaded_image = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image.resize((150, 150)), caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button('🔍 Classify'):
                with st.spinner('Classifying...'):
                    prediction, confidence = predict_image_class(model, uploaded_image, class_indices)

                if prediction is not None:
                    st.success(f'Prediction: **{prediction}**')
                    st.write(f'Confidence: **{confidence:.2f}%**')

                    if "healthy" in prediction.lower():
                        st.info("✅ Your plant looks healthy! No action needed.")
                    else:
                        st.warning("⚠️ This plant might have a disease. Consider checking for symptoms.")
    except UnidentifiedImageError:
        st.error("❌ Error: Uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")




