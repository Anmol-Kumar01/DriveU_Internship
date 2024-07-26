#Author : Anmol Kumar (Intern @ DriveU)
#Date   : 1 July 2024



#pip install torch
#pip install ultralytics
#pip install opencv-python

import cv2
import torch
from PIL import Image

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_car(image_path):
    """
    Detects if a car is present in the image.
    Args:  image_path (str): The path to the image file.
    Returns:  bool: True if a car is detected, False otherwise.
    
    """
    model.eval()
    image = Image.open(image_path)
    results = model(image)
    for result in results.xyxy[0]:
        if result[5] in [2, 3, 5, 6, 7]:  
            return True
    return False


# Example 

image_path = r"C:\Users\Anmol Kumar\Desktop\DriveU\filtered_images\drop_car_front_6825.jpg"


result = detect_car(image_path)
if result:
    print("The image contains a car.")
else:
    print("The image does not contain a car.")
