from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

damage_model = YOLO("models/damage.pt")
clean_model = YOLO("models/clean.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    damage_results = damage_model(image)
    damage_pred = damage_results.pandas().xywhn[0].to_dict(orient="records")
    clean_results = clean_model(image)
    clean_pred = clean_results.pandas().xywhn[0].to_dict(orient="records")
    return {
        "damage": damage_pred,
        "clean": clean_pred
    }
