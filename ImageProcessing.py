# Author : Anmol Kumar ( Intern at DriveU )
# Date : 15 July 2024



import cv2
import numpy as np
from PIL import Image
import torch
from torch.hub import load as hub_load
from fastapi import FastAPI
import requests

# Load YOLOv5 model only at once 
model = hub_load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class ImageProcessing(object):
    """
    A class to perform various image processing tasks including vehicle detection,
    noise estimation, sharpness estimation, brightness estimation, contrast estimation,
    saturation estimation, image analysis, and image status determination.
    """
    
    # # Initialize with an image path
    def __init__(self, image_path):
        
        """
        Initialize the class with the path to an image .

        Args:
            image_path (str): Path to the image file.
        """
        
        self.image_path = image_path
        self.image = self.load_image_from_url(image_path)
        if self.image is None:
            raise ValueError(f"Error: Unable to load image from URL {image_url}")
        
        
    def load_image_from_url(self, url):
        """
        Download the image from the from url
        
        Args:
            url (str): URL of the image to download.
        """
        
        # Download image from URL
        response = requests.get(url)
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Error: Unable to load image from URL {url}")
        return image
    
    
    def detect_vehicle(self):
        """
        Detect vehicles in the loaded image using a pre-trained YOLOv5 model.

        Returns:
            bool: True if a vehicle is detected, False otherwise.
        """
        
        # Set the model to evaluation mode
        model.eval()

        # Convert OpenCV BGR image to RGB (PIL expects RGB)
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Convert to PIL image
        pil_image = Image.fromarray(image_rgb)

        # Detect objects in the image
        results = model(pil_image)

        # Check if any vehicle is detected
        for result in results.xyxy[0]:
            if result[5] in [2, 3, 5, 6, 7]:
                return True
        return False
    

    # Estimate noise level in the image
    def estimate_noise(self):
        print("Dafad")
        """
        Estimate the noise level in the image.

        Returns:
            float: Estimated noise level percentage.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        stddev = np.std(gray)
        noise = stddev / mean * 100
        noise = round(noise)
        return noise
    

    # Estimate sharpness of the image
    def estimate_sharpness(self):
        """
        Estimate the sharpness of the image.

        Returns:
            float: Estimated sharpness value.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness
    

    # Estimate brightness of the image
    def estimate_brightness(self):
        """
        Estimate the brightness of the image.

        Returns:
            float: Estimated brightness value.
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2].mean()
        return brightness
    

    # Estimate contrast of the image
    def estimate_contrast(self):
        """
        Estimate the contrast of the image.

        Returns:
            float: Estimated contrast value.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        return contrast
    

    # Estimate mean saturation of the image
    def saturation(self):
        """
        Estimate the mean saturation of the image.

        Returns:
            float: Estimated mean saturation value.
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = saturation.mean()
        return mean_saturation
    

    @staticmethod
    def normalize(value, max_value):
        """
        Normalize a value based on a maximum value.

        Args:
            value (float): Value to normalize.
            max_value (float): Maximum possible value.

        Returns:
            float: Normalized value between 0 and 100.
        """
        return min(max((value / max_value) * 100, 0), 100)
    

    # Get image resolution (width, height)
    def image_resolution(self):
        """
        Get the resolution (width, height) of the image.

        Returns:
            tuple: Image width and height.
        """
        height, width, _ = self.image.shape
        return width, height
    

    # Analyze various image parameters and return a dictionary
    def image_params(self):
        """
        Analyze various parameters of the image including noise, sharpness,
        brightness, contrast, and saturation.

        Returns:
            dict: Dictionary containing image analysis results.
        """
        noise_level = self.estimate_noise()
        sharpness = self.estimate_sharpness()
        brightness = self.estimate_brightness()
        contrast = self.estimate_contrast()
        mean_saturation = self.saturation()

        # Maximum possible values for normalization
        max_noise_value = 250
        max_sharpness_value = 1000
        max_contrast_value = 100
        max_brightness_value = 160
        max_mean_saturation = 100

        # Normalize parameters
        noise_percentage = self.normalize(noise_level, max_noise_value)
        sharpness_percentage = self.normalize(sharpness, max_sharpness_value)
        brightness_percentage = self.normalize(brightness, max_brightness_value)
        contrast_percentage = self.normalize(contrast, max_contrast_value)
        saturation_percentage = self.normalize(mean_saturation, max_mean_saturation)
        image_resolutions = self.image_resolution()

        result = {
            'resolution': image_resolutions,
            'noise': int(noise_percentage),
            'sharpness': int(sharpness_percentage),
            'brightness': int(brightness_percentage),
            'contrast': int(contrast_percentage),
            'saturation': int(saturation_percentage)
        }
        return result
    

    # Determine if the image is clear based on image parameters
    def is_image_clear(self, image_params):
        """
        Determine if the image is clear based on specified image parameters.

        Args:
            image_params (dict): Dictionary containing image parameters.

        Returns:
            bool: True if the image is clear, False otherwise.
        """
        image_status = True

        if image_params['brightness'] < 20 and image_params['noise'] > 75:
            image_status = False
        if image_params['sharpness'] < 10 and image_params['brightness'] < 20 and image_params['saturation'] > 50:
            image_status = False
        if image_params['sharpness'] < 10 and image_params['noise'] > 20 and image_params['saturation'] > 50:
            image_status = False
        elif image_params['sharpness'] < 30:
            if image_params['brightness'] < 40:
                if image_params['saturation'] > 50 and image_params['noise'] > 30:
                    image_status = False
        elif image_params['sharpness'] < 50:
            if image_params['brightness'] < 40:
                if image_params['saturation'] > 60 and image_params['contrast'] < 40:
                    image_status = False
            elif image_params['noise'] > 20 and image_params['saturation'] > 60 and image_params['contrast'] < 40:
                image_status = False
        elif image_params['sharpness'] < 50 and (image_params['noise'] > 50 or image_params['brightness'] < 25):
            image_status = False
        elif image_params['brightness'] <= 20 and image_params['sharpness'] < 20:
            image_status = False
        elif image_params['saturation'] > 85 and image_params['sharpness'] < 5:
            image_status = False
        return image_status
    

    # Determine the status (Good, Average, Below Average, Bad) of the image
    def image_status(self, image_params):
        """
        Determine the status (Good, Average, Below Average, Bad) of the image
        based on specified image parameters.

        Args:
            image_params (dict): Dictionary containing image parameters.

        Returns:
            str: Status of the image.
        """
        image_status_list = ["Good", "Average", "Below Average", "Bad"]
        image_status = None

        if not self.is_image_clear(image_params):
            image_status = image_status_list[3]
        elif image_params['sharpness'] > 75 and image_params['brightness'] > 50:
            if image_params['noise'] < 20:
                image_status = image_status_list[0]
            elif image_params['saturation'] < 30:
                image_status = image_status_list[0]
            elif image_params['contrast'] > 50:
                image_status = image_status_list[0]
            else:
                image_status = image_status_list[1]

        elif image_params['sharpness'] <= 50:
            if image_params['brightness'] < 45:
                if image_params['noise'] > 20 and image_params['saturation'] > 50:
                    image_status = image_status_list[2]
                elif image_params['noise'] > 35 and image_params['sharpness'] < 15:
                    image_status = image_status_list[2]
                else:
                    image_status = image_status_list[1]
            elif image_params['sharpness'] < 10 and image_params['noise'] > 20:
                if image_params['saturation'] > 30:
                    image_status = image_status_list[2]
                else:
                    image_status = image_status_list[1]
            elif image_params['noise'] >= 20:
                if image_params['saturation'] > 50 and image_params['contrast'] < 60:
                    image_status = image_status_list[2]
                elif image_params['sharpness'] <= 25 and image_params['contrast'] < 40:
                    image_status = image_status_list[2]
                else:
                    image_status = image_status_list[1]
            elif image_params['sharpness'] < 20:
                if image_params['sharpness'] < 5:
                    image_status = image_status_list[2]
                else:
                    image_status = image_status_list[1]
            else:
                image_status = image_status_list[0]

        elif image_params['sharpness'] >= 60:
            if image_params['noise'] > 20 and image_params['saturation'] < 60:
                image_status = image_status_list[1]
            elif image_params['noise'] > 35:
                image_status = image_status_list[2]
            else:
                image_status = image_status_list[0]

        elif image_params['sharpness'] < 20:
            if image_params['brightness'] < 30:
                if image_params['saturation'] > 30:
                    if image_params['contrast'] < 40:
                        image_status = image_status_list[2]
                else:
                    image_status = image_status_list[1]
        else:
            image_status = image_status_list[0]

        return image_status
    

    # Get detailed image analysis including status, vehicle detection, and clarity
    def get_image_details(self):
        """
        Get detailed analysis of the image including image parameters,
        image status, vehicle detection, and clarity.

        Returns:
            dict: Dictionary containing detailed image analysis results.
        """
        image_params = self.image_params()
        image_clear = self.is_image_clear(image_params)
        status = self.image_status(image_params)
        vehicle_detected = self.detect_vehicle()

        image_params['status'] = status
        image_params['vehicle'] = vehicle_detected
        image_params['clear'] = image_clear
        return image_params



# FastAPI application setup
app = FastAPI()

# Initialize ImageProcessing class
image_processor = None

@app.get("/image_details/")
def analyze_image(image_url: str):
    global image_processor
    try:
        image_processor = ImageProcessing(image_url)
        details = image_processor.get_image_details()
        return details
    except Exception as e:
        return {"error": str(e)}

@app.get("/image_params/")
def image_params(image_url: str):
    global image_processor
    try:
        image_processor = ImageProcessing(image_url)
        details = image_processor.image_params()
        return details
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/vehicle/")
def vehicle(image_url: str):
    global image_processor
    try:
        image_processor = ImageProcessing(image_url)
        details = image_processor.detect_vehicle()
        return details
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/is_clear/")
def image_clear(image_url: str):
    global image_processor
    try:
        image_processor = ImageProcessing(image_url)
        details = image_processor.image_params()
        clear = image_processor.is_image_clear(details)
        return clear
    except Exception as e:
        return {"error": str(e)}

    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)