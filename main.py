from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from lib.prueba_modelo import test_concepto

app = FastAPI()

class Params(BaseModel):
    smiles: List[str] = []

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/modelo")
def update_item(params: Params):
    df = test_concepto(params.smiles)
    return {"smiles": params.smiles,
            "prediction": list(df.prediction),
            "probabilities": list(df.prediction_prob)}