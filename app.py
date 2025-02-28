import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
import torch

# Load models
@st.cache_resource
def load_graffiti_model():
    return YOLO("best.pt")

graffiti_model = load_graffiti_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_road_masking_model():
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    return processor, model

processor, road_masking_model = load_road_masking_model()

@st.cache_resource
@st.cache_resource
@st.cache_resource
def load_road_classifier():
    try:
        road_classifier = torch.load("best_model_binary.pth", map_location=device)
        if isinstance(road_classifier, dict):  # If it's a state_dict, load it into a model
            from torchvision import models
            road_classifier_model = models.resnet50(pretrained=False)  # Ensure correct architecture
            road_classifier_model.fc = torch.nn.Linear(road_classifier_model.fc.in_features, 1)
            road_classifier_model.load_state_dict(road_classifier)
            road_classifier_model.to(device)
            return road_classifier_model
        road_classifier.to(device)
        return road_classifier
    except Exception as e:
        st.error(f"Error loading road classifier model: {e}")
        return None



road_classifier = load_road_classifier()

def predict_image(image):
    image_cv = np.array(image)
    results = graffiti_model(image_cv)
    return results[0].plot()

def mask_road(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = road_masking_model(**inputs)
    logits = outputs.logits.squeeze().cpu().numpy()
    mask = np.argmax(logits, axis=0) == 0
    return mask

def classify_road(image):
    image_resized = cv2.resize(np.array(image), (224, 224))  # Resize to match model input
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = road_classifier(image_tensor)
    
    prediction = torch.sigmoid(output).item()
    return "Bad Road" if prediction > 0.5 else "Good Road"


# Streamlit UI
st.title("Google Street View Disorder Detection")

# Dropdown menu for selecting model
option = st.sidebar.selectbox("Choose detection type", ["Graffiti", "Bad Roads"])

if option == "Graffiti":
    st.header("Graffiti Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        result_image = predict_image(image)
        st.image(result_image, caption="Predicted Image with Bounding Boxes", use_container_width=True)

elif option == "Bad Roads":
    st.header("Bad Roads Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        road_mask = mask_road(image)
        road_mask_resized = cv2.resize(road_mask.astype(np.uint8), (image.width, image.height), interpolation=cv2.INTER_NEAREST)
        road_only_image = np.array(image) * road_mask_resized[:, :, np.newaxis]
        road_condition = classify_road(road_only_image)
        st.write(f"Road Condition: **{road_condition}**")
