import torch
import redis
import json
import os
from transformers import pipeline
from fastapi import FastAPI
from dotenv import load_dotenv
from elasticapm.contrib.starlette import make_apm_client, ElasticAPM
from datetime import datetime, timezone

load_dotenv()

apm_config = {
 'SERVICE_NAME': 'SentimentAnalysisApp',
 'SERVER_URL': os.getenv("APM_SERVER"),
 'SECRET_TOKEN': os.getenv("APM_TOKEN"),
 'ENVIRONMENT': 'dev',
 'GLOBAL_LABELS': 'platform=FastAPI, application=SentimentAnalysisApp'
}
apm = make_apm_client(apm_config)

app = FastAPI()
app.add_middleware(ElasticAPM, client=apm)
sentiment = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    top_k=1
)

redis_url = os.getenv("REDIS_URL", "localhost")
redis_port = os.getenv("REDIS_PORT", 6379)
r = redis.Redis(host=redis_url, port=redis_port)

start_date = datetime.now(timezone.utc)
cur_version = os.getenv("VERSION")

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

@app.get("/stats")
def get_stats() -> dict:
    return {
        "start_date" : start_date,
        "version" : cur_version
    }
    