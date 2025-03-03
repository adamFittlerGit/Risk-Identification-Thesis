import torch
from ultralytics import YOLO

def load_yolo_model(weights_path):
    # Load the YOLO model with the specified weights
    model = YOLO(weights_path)
    return model

def process_image(model, image_path):
    # Process the image and return the detected entities
    results = model(image_path)
    return results 