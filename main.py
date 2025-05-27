from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
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

# 예외 핸들러 추가 (입력 검증 실패 시 에러 상세 반환)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body  # 요청 본문 전체도 반환
        }
    )

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

    return req.songs[best_index]

