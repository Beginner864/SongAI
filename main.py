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

# Lemmatization을 위한 NLTK
import nltk
nltk.data.path.append("./nltk_data")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import TreebankWordTokenizer

# 빠른 토크나이저 (punkt 회피)
tokenizer = TreebankWordTokenizer()

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )

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

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    words = tokenizer.tokenize(text)
    pos_tags = pos_tag(words)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmas)

# 감정 입력 정제 함수: 영어만 lemmatization 적용
def clean_korean_mood(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = text.strip().lower()
    return lemmatize_text(text)

# JSON 데이터 로드 및 벡터화 학습
with open("songs.json", "r", encoding="utf-8") as f:
    all_songs = json.load(f)

corpus = [clean_korean_mood(f"{song['mood']} {song['title']}") for song in all_songs]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.user_songs:
        return {"error": "사용자 보유 곡이 없습니다."}
    if not req.mood:
        return {"error": "기분 입력이 필요합니다."}

    cleaned = clean_korean_mood(req.mood)
    user_input_vector = vectorizer.transform([cleaned])

    user_corpus = [clean_korean_mood(f"{song.mood} {song.title}") for song in req.user_songs]
    user_vectors = vectorizer.transform(user_corpus)
    similarities = cosine_similarity(user_input_vector, user_vectors)[0]

    print(f"\n[INPUT MOOD] \"{req.mood}\"\n→ cleaned → \"{cleaned}\"\n")
    print("[SIMILARITIES] 사용자 보유 곡 중 유사도 목록 (유사도 내림차순):")

    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    candidates = []

    for i, sim in ranked:
        song = req.user_songs[i]
        print(f"  - ID {song.id:>3} | {sim:.4f} | {song.title}")
        if sim >= 0.05:
            candidates.append(song)

    if not candidates:
        print("유사도 기준(0.05) 이상인 곡이 없음? 체감상 0.95이상인데..\n")
        return {"error": "추천 가능한 곡이 없습니다."}

    print("유사도 기준 통과 → 랜덤 추천 진행\n")
    return random.choice(candidates)
