import json
import re
import random
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

app = FastAPI()

# --- 예외 핸들러 (요청 유효성 검증 실패 시 422 반환) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )

# --- 모델 정의 ---
class Song(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    streaming_url: str

class RecommendRequest(BaseModel):
    mood: str
    songs: List[Song]  # 사용자가 가진 곡 목록

# --- 감정 정제 함수 ---
def clean_korean_mood(text: str) -> str:
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)  # 한글, 영문, 공백만 허용
    text = re.sub(r"(.)\1{2,}", r"\1")  # 반복 문자 줄이기
    return text.strip().lower()

# --- 서버 시작 시 songs.json 로딩 및 다중 feature 벡터화 ---
with open("songs.json", "r", encoding="utf-8") as f:
    all_songs = json.load(f)

# ✅ 다중 feature: mood + genre + title
corpus = [
    f"{song['mood']} {song['genre']} {song['title']}" for song in all_songs
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)  # 전체 곡에 대해 벡터화 (고정)

# --- 추천 엔드포인트 ---
@app.post("/recommend")
def recommend(req: RecommendRequest):
    cleaned = clean_korean_mood(req.mood)
    user_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vec, X)[0]

    # 유사도 내림차순 정렬
    sorted_indices = np.argsort(similarities)[::-1]

    # 사용자가 보유한 곡 ID만 필터링
    user_song_ids = {song.id for song in req.songs}

    for idx in sorted_indices:
        candidate = all_songs[idx]
        if candidate["id"] in user_song_ids:
            return candidate  # 가장 유사한 곡 1개 추천

    return {"error": "사용자 보유 곡 중 추천 가능한 곡이 없습니다."}
