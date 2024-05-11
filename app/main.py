import torch
import redis
import json
from transformers import pipeline
from fastapi import FastAPI


app = FastAPI()

sentiment = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

r = redis.Redis(host="redis", port=6379)
    
@app.get("/is_cuda")
def is_cuda():
    return {"CUDA": torch.cuda.is_available()}

@app.post("/predict")
def analyse(to_analyse: str) -> list[dict]:
    preds = sentiment(to_analyse)[0] 
    r.set(to_analyse, json.dumps(preds))
    return preds

@app.get("/predict_history")
def get_predict_history():
    keys = r.keys('*')
    all_strings = []
    for k in keys:
        all_strings.append({k: json.loads(r.get(k))})
    return all_strings