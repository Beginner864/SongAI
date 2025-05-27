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

# 예외 핸들러: 요청 검증 실패 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )

# 데이터 모델 정의
class Song(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    streaming_url: str

class RecommendRequest(BaseModel):
    mood: str
    user_songs: List[Song]

# 감정 입력 정제 함수
def clean_korean_mood(text: str) -> str:
    if not text:
        return ""  # 비어 있는 경우는 빈 문자열을 반환
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)  # 한글과 영어 외의 문자는 제거
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # 같은 문자 반복 제거 (예: cooool → cool)
    return text.strip().lower()


# 서버 시작 시 songs.json을 이용해 TF-IDF 벡터라이저 학습
with open("songs.json", "r", encoding="utf-8") as f:
    all_songs = json.load(f)

# mood, title로만 백터화
corpus = [f"{song['mood']} {song['title']}" for song in all_songs]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)  # 단어 어휘 사전 학습 (vectorizer 재사용)

# 추천 엔드포인트
@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.user_songs:
        return {"error": "사용자 보유 곡이 없습니다."}
    if not req.mood:
        return {"error": "기분 입력이 필요합니다."}

    cleaned = clean_korean_mood(req.mood)
    user_input_vector = vectorizer.transform([cleaned])

    # 사용자 보유 곡 텍스트 벡터화
    user_corpus = [f"{song.mood} {song.title}" for song in req.user_songs]
    user_vectors = vectorizer.transform(user_corpus)

    similarities = cosine_similarity(user_input_vector, user_vectors)[0]

    print(f"\n[INPUT MOOD] \"{req.mood}\"\n→ cleaned → \"{cleaned}\"\n")
    print("[SIMILARITIES] 사용자 보유 곡 중 유사도 목록 (유사도 내림차순):")

    # 유사도 기준 정렬하여 로그 출력
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    candidates = []
    for i, sim in ranked:
        song = req.user_songs[i]
        print(f"  - ID {song.id:>3} | {sim:.4f} | {song.title}")
        if sim >= 0.5:  # 0.5 이상인 곡만 필터링
            candidates.append(song)

    if not candidates:
        print("유사도 기준(0.5) 이상인 곡이 없음\n")
        return {"error": "추천 가능한 곡이 없습니다."}

    print("유사도 기준 통과 → 랜덤 추천 진행\n")
    return random.choice(candidates)
