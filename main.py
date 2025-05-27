from fastapi import FastAPI, Query
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# 샘플 노래 데이터 불러오기
with open("songs.json", "r", encoding="utf-8") as f:
    songs = json.load(f)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([song["mood"] for song in songs])

@app.get("/recommend")
def recommend(mood: str):
    user_vec = vectorizer.transform([mood])
    similarities = cosine_similarity(user_vec, X)
    best_index = similarities.argmax()
    return songs[best_index]
