import os
import uvicorn
from fastapi import FastAPI
from actor_face_movie_util import router as actor_face_movie_router
from emotion_music_movie_util import router as emotion_music_movie_router
from worlds_subtitle_movie import router as worlds_subtitle_movie_router

app = FastAPI()

# Include routers from other modules
app.include_router(actor_face_movie_router, prefix="/actor_face_movie")
app.include_router(emotion_music_movie_router, prefix="/emotion_music_movie")
app.include_router(worlds_subtitle_movie_router, prefix="/worlds_subtitle_movie")

if __name__ == "__main__":
    print("서버를 시작합니다...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\ICT05_04\\Desktop\\finalproject\\movie\\translate-movie-427703-adec2ac5235a.json"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
