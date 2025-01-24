import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load the YOLO model (using the path to your model)
model = YOLO('best.pt')  # Path to your saved YOLO model

# Function to make predictions and draw bounding boxes
def predict_image(image):
    # Convert the uploaded image to a format compatible with YOLO model (OpenCV format)
    image_cv = np.array(image)
    
    # Run the model on the image
    results = model(image_cv)
    
    # Extract the first result (if there are multiple outputs, take the first one)
    result = results[0]

    # Get the image with bounding boxes drawn
    output_image = result.plot()  # This will automatically draw the bounding boxes

    return output_image

# Streamlit UI
st.title("Google Street View Disorder Detection")
st.write("Upload an image of a street, and our model will detect disorder.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction and display the result
    result_image = predict_image(image)
    
    # Display the result
    st.image(result_image, caption="Predicted Image with Bounding Boxes", use_column_width=True)
