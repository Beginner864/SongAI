from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from typing import List
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 데이터 불러오기
with open("songs.json", "r", encoding="utf-8") as f:
    songs = json.load(f)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([song["mood"] for song in songs])

class MoodRequest(BaseModel):
    mood: str

@app.post("/recommend")
def recommend(req: MoodRequest):
    user_vec = vectorizer.transform([req.mood])
    similarities = cosine_similarity(user_vec, X)
    best_index = similarities.argmax()
    return songs[best_index]
