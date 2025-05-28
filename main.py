import json
import re
import random
import numpy as np
import os
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# NLTK 설정
import nltk
nltk.data.path.append("./nltk_data")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import TreebankWordTokenizer

# GPT API 설정
import openai

#로컬에서 하면
#from dotenv import load_dotenv
#load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경변수 사용 권장

# FastAPI 앱 초기화
app = FastAPI()

# 예외 처리 핸들러
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
    user_songs: List[Song]

# NLP 도구
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    elif tag.startswith("V"): return wordnet.VERB
    elif tag.startswith("N"): return wordnet.NOUN
    elif tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def lemmatize_text(text):
    words = tokenizer.tokenize(text)
    pos_tags = pos_tag(words)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmas)

# 정제 함수
def clean_korean_mood(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = text.strip().lower()
    return lemmatize_text(text)

# GPT 기반 한글 → 영어 번역 + 캐시
mood_translation_cache = {}

def gpt_translate_to_english(korean_mood: str) -> str:
    if korean_mood in mood_translation_cache:
        return mood_translation_cache[korean_mood]

    prompt = f"""Translate the following mood word from Korean to English.
Return only a single English word that best matches the mood.

Korean: {korean_mood}
English:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10,
        )
        english_mood = response["choices"][0]["message"]["content"].strip().lower()
        mood_translation_cache[korean_mood] = english_mood
        return english_mood
    except Exception as e:
        print(f"[GPT ERROR] '{korean_mood}' 번역 실패: {e}")
        return korean_mood  # 실패 시 원문 그대로 사용

# 학습 데이터 로드 및 벡터화
with open("songs.json", "r", encoding="utf-8") as f:
    all_songs = json.load(f)

corpus = [clean_korean_mood(f"{song['mood']} {song['title']}") for song in all_songs]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# 추천 API
@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.user_songs:
        return {"error": "사용자 보유 곡이 없습니다."}
    if not req.mood:
        return {"error": "기분 입력이 필요합니다."}

    # 입력 기분 정제
    cleaned = clean_korean_mood(req.mood)
    user_input_vector = vectorizer.transform([cleaned])

    # 사용자 곡 정제 + 번역
    user_corpus = []
    for song in req.user_songs:
        mood_translated = gpt_translate_to_english(song.mood)
        cleaned_text = clean_korean_mood(f"{mood_translated} {song.title}")
        user_corpus.append(cleaned_text)

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
        print("유사도 기준(0.05) 이상인 곡이 없음")
        return {"error": "추천 가능한 곡이 없습니다."}

    print("유사도 기준 통과 → 랜덤 추천 진행\n")
    return random.choice(candidates)
