import torch
import redis
import json
from transformers import pipeline
from fastapi import FastAPI


app = FastAPI()

sentiment = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    top_k=1
)

r = redis.Redis(host="redis", port=6379)

@app.get("/is_cuda")
def is_cuda() -> bool:
    return torch.cuda.is_available()

@app.post("/predict")
def analyse(to_analyse: str) -> dict:
    preds = sentiment(to_analyse)[0] 
    r.set(to_analyse, json.dumps(preds))
    return preds[0]

@app.get("/predict_history")
def get_predict_history() -> list[dict]:
    keys = r.keys('*')
    all_strings = []
    for k in keys:
        all_strings.append({k: json.loads(r.get(k))})
    return all_strings