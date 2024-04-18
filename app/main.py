from fastapi import FastAPI
from app.root import read_root
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import os

app = FastAPI()


class Data(BaseModel):
    imageArr: list[str] = []


@app.get("/")
def read_main():
    return read_root()


@app.post("/predict")
async def predict(data: Data):
    HOME = os.getcwd()
    namrArr = []
    imageArr = data.imageArr
    modelPath = HOME+"\\Model\\best.pt"
    model = YOLO(modelPath)

    for img in imageArr:
        results = model(img)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        namrArr.append(names_dict[np.argmax(probs)])

    resultsName = max(set(namrArr), key=namrArr.count)
    return {
        "resultsName" : resultsName 
    }
