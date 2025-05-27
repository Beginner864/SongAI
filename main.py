from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import re

app = FastAPI()

# 곡 모델
class Song(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    streaming_url: str

# 요청 모델
class RecommendRequest(BaseModel):
    mood: str
    songs: List[Song]

# 한국어 감정 정제 함수
def clean_korean_mood(text: str) -> str:
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)  # 한글, 영문, 공백만
    text = re.sub(r"(.)\1{2,}", r"\1")  # 반복 문자 축소
    return text.strip().lower()

@app.post("/recommend")
def recommend(req: RecommendRequest):
    cleaned = clean_korean_mood(req.mood)

    if not req.songs:
        return {"error": "추천할 곡이 없습니다."}

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([song.mood for song in req.songs])
    user_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vec, X)[0]

    top_n = min(5, len(similarities))
    top_indices = np.argsort(similarities)[-top_n:]
    best_index = int(random.choice(top_indices))

    song = req.songs[best_index]
    return song
