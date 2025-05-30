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

# OpenAI 최신 API
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI 앱 초기화
app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )

# 모델 정의
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

# NLP 구성
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    elif tag.startswith("V"): return wordnet.VERB
    elif tag.startswith("N"): return wordnet.NOUN
    elif tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

# 단어 1개일 경우 강제로 동사 처리
def smart_lemmatize(text: str) -> str:
    words = tokenizer.tokenize(text)
    if len(words) == 1:
        return lemmatizer.lemmatize(words[0], wordnet.VERB)
    else:
        pos_tags = pos_tag(words)
        lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        return " ".join(lemmas)

def clean_korean_mood(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = text.strip().lower()
    return text

# GPT 번역 캐시
mood_translation_cache = {}

def is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))

def gpt_translate_to_english(mood: str) -> str:
    if mood in mood_translation_cache:
        print(f"[CACHE] {mood} → {mood_translation_cache[mood]}")
        return mood_translation_cache[mood]

    if not is_korean(mood):
        print(f"[SKIP] 이미 영어로 판단된 감정: {mood}")
        return mood.lower()

    prompt = f"""Translate the following Korean mood expression to a single English adjective or verb (not a noun) that best expresses the emotion.
ONLY return one English word, and NOTHING ELSE. Do not include explanations or synonyms.

Korean: {mood}
English:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10,
        )
        english_mood = response.choices[0].message.content.strip().lower().split()[0]
        print(f"[GPT] 번역 성공: {mood} → {english_mood}")
        mood_translation_cache[mood] = english_mood
        return english_mood
    except Exception as e:
        print(f"[GPT ERROR] '{mood}' 번역 실패: {e}")
        return mood  # 실패 시 원본 그대로 사용

# 학습용 벡터화
with open("songs.json", "r", encoding="utf-8") as f:
    all_songs = json.load(f)

corpus = [clean_korean_mood(f"{smart_lemmatize(song['mood'])} {smart_lemmatize(song['mood'])} {song['title']}") for song in all_songs]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.user_songs:
        return {"error": "사용자 보유 곡이 없습니다."}
    if not req.mood:
        return {"error": "기분 입력이 필요합니다."}

    translated = gpt_translate_to_english(req.mood)
    lemmatized = smart_lemmatize(translated)
    cleaned = clean_korean_mood(f"{lemmatized} {lemmatized}")
    print(f"[INPUT] 사용자가 입력한 기분: {req.mood} → translated: {translated} → lemmatized: {lemmatized} → cleaned: {cleaned}")

    user_input_vector = vectorizer.transform([cleaned])

    user_corpus = []
    for song in req.user_songs:
        mood_translated = gpt_translate_to_english(song.mood)
        mood_lemmatized = smart_lemmatize(mood_translated)
        cleaned_text = clean_korean_mood(f"{mood_lemmatized} {mood_lemmatized} {song.title}")
        print(f"[SONG] {song.title} | mood: {song.mood} → {mood_translated} → lemmatized: {mood_lemmatized} → cleaned: {cleaned_text}")
        user_corpus.append(cleaned_text)

    user_vectors = vectorizer.transform(user_corpus)
    similarities = cosine_similarity(user_input_vector, user_vectors)[0]

    print("[SIMILARITY] 유사도:")
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    candidates = []

    for i, sim in ranked:
        song = req.user_songs[i]
        print(f"  - {song.title} (ID {song.id}) | 유사도: {sim:.4f}")
        if sim >= 0.05:
            candidates.append(song)

    if not candidates:
        print("유사도 기준(0.05) 이상인 곡이 없음")
        return {"error": "추천 가능한 곡이 없습니다."}

    print("유사도 기준 통과 → 랜덤 추천 실행")
    return random.choice(candidates)


