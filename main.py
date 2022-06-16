from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from prueba_modelo import test_concepto
import pandas as pd
import json

app = FastAPI()

class Params(BaseModel):
    smiles: List[str] = []

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/modelo")
def update_item(params: Params):
    df = test_concepto(params.smiles)
    result = df.to_json(orient="table")
    parsed = json.loads(result)
    result = json.dumps(parsed, indent=4)
    return parsed