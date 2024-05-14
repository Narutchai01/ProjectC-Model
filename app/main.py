from fastapi import FastAPI
from app.root import read_root
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()



def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


class Data(BaseModel):
    imageArr: list[str] = []


@app.get("/")
def read_main():
    return read_root()


import urllib.request

@app.post("/predict")
async def predict(data: Data):

    imageArr = data.imageArr
    namrArr = []
    modelPath = os.path.join("app","Model","run18","best.pt")
    model = YOLO(modelPath)

    for img_url in imageArr:
        
        img = get_image(img_url)

        results = model.predict(img, stream=False)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        namrArr.append(names_dict[np.argmax(probs)])

    resultsName = max(set(namrArr), key=namrArr.count)
    resultsName = resultsName.split("___")
    NameDisease = resultsName[1].replace("_", " ")

    return {
        "NamePlant": resultsName[0],
        "NameDisease": NameDisease
    }
