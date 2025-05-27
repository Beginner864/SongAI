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

# 예외 핸들러: 잘못된 요청 본문 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )

# 데이터 모델
class Song(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    streaming_url: str

class RecommendRequest(BaseModel):
    mood: str
    songs: List[Song]  # 사용자 보유 곡

# 감정 문자열 정제
def clean_korean_mood(text: str) -> str:
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text.strip().lower()


# 서버 시작 시 songs.json 로딩
with open("songs.json", "r", encoding="utf-8") as f:
    all_songs = json.load(f)

# 전체 곡 코퍼스 벡터화
corpus = [f"{song['mood']} {song['genre']} {song['title']}" for song in all_songs]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 추천 엔드포인트
@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.mood:
        return {"error": "mood is required."}

    cleaned = clean_korean_mood(req.mood)
    user_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vec, X)[0]

    user_song_ids = {song.id for song in req.songs}

    print(f"\n[INPUT MOOD] \"{req.mood}\" → cleaned → \"{cleaned}\"\n")
    print("[SIMILARITIES] 사용자 보유 곡 중 유사도 목록:")

    candidates = []
    for i, sim in enumerate(similarities):
        song = all_songs[i]
        if song["id"] in user_song_ids:
            print(f"  - ID {song['id']:>3} | {sim:.4f} | {song['title']}")
            if sim >= 0.7:
                candidates.append(song)

    if not candidates:
        print("유사도 기준(0.7) 이상인 곡이 없음\n")
        return {"error": "추천 가능한 곡이 없습니다."}

    print("유사도 기준 통과 → 랜덤 추천 진행\n")
    return random.choice(candidates)
