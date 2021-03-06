from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from apply_modelos import test_concepto
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
    result = test_concepto(params.smiles)
    return result