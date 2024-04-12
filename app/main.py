from fastapi import FastAPI
from app.root import read_root
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    arrIMG: list[str] = []



@app.get("/")
def read_main():
    return read_root()


@app.post("/predict")
async def predict(data : Data):
    return {"data": data.arrIMG}