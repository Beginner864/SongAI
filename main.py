from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# 데이터 로딩
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
    song = songs[best_index]

    # 필요한 필드만 응답
    return {
        "id": song.get("id", 0),
        "title": song.get("title", ""),
        "artist": song.get("artist", ""),
        "genre": song.get("genre", "Unknown"),
        "mood": song.get("mood", ""),
        "streaming_url": song.get("streaming_url", "")
    }
