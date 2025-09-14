from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()


damage_model = YOLO("models/damage.pt")
clean_model = YOLO("models/clean.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        damage_results = damage_model(image_array)[0]
        if damage_results.boxes:
            top_damage = damage_results.boxes[0]
            damage_class = int(top_damage.cls.cpu().numpy()[0])
        else:
            damage_class = 0

        clean_results = clean_model(image_array)[0]
        if clean_results.boxes:
            top_clean = clean_results.boxes[0]
            clean_class = int(top_clean.cls.cpu().numpy()[0])
        else:
            clean_class = 0

        return {"damage": damage_class, "cleanliness": clean_class}

    except Exception as e:
        return {"error": str(e)}
