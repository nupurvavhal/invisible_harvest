from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

# Setup templates and static files
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load your trained model
MODEL = tf.keras.models.load_model(str(BASE_DIR / "model" / "fixedfruit_model_v2.keras"))
CLASS_NAMES = ['Apple_Fresh', 'Apple_Rotten', 'Banana_Fresh', 'Banana_Rotten', 
               'Orange_Fresh', 'Orange_Rotten', 'Strawberry_Fresh', 'Strawberry_Rotten']

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_preprocessed = preprocess_image(contents)
    
    predictions = MODEL.predict(img_preprocessed)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    label = CLASS_NAMES[class_idx]
    
    # Extract fruit and status
    fruit, status = label.split('_')
    
    return {
        "fruit": fruit,
        "status": status,
        "confidence": f"{score*100:.2f}%",
        "eatability": "Safe ✅" if status == "Fresh" else "Avoid ❌",
        "shelf_life": "5-7 Days" if status == "Fresh" else "Expired",
        "advice": "Keep in fridge." if status == "Fresh" else "Compost it."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)