from fastapi import FastAPI
from app.root import read_root
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import os

app = FastAPI()


# class Data(BaseModel):
#     arrIMG: list[str] = []


@app.get("/")
def read_main():
    return read_root()


@app.post("/predict")
async def predict():
    HOME = os.getcwd()
    modelPath = "./app/Model/best.pt"
    model = YOLO(modelPath)

    imageArr = [
        "./app/testIMG/0cashew_valid_anthracnose.JPG",
        "./app/testIMG/1cashew_valid_anthracnose.JPG",
        "./app/testIMG/2cashew_valid_anthracnose.JPG",
        "./app/testIMG/34cashew_valid_red rust.JPG",
    ]

    results = model("./app/testIMG/0cashew_valid_anthracnose.JPG")

    names_dict = results[0].names

    probs = results[0].probs.data.tolist()

    return {
    "predict": names_dict[np.argmax(probs)]
    }
