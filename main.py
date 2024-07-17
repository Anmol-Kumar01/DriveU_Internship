from fastapi import FastAPI
from ImageProcessing import ImageProcessing 
import uvicorn
# FastAPI application setup
app = FastAPI()

# Initialize ImageProcessing class
image_processor = None

@app.get("/analyze_image/")
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