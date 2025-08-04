from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

# Sample training data
mood_data = {
    "happy": ["I feel great today!", "Life is beautiful", "This is the best day ever"],
    "sad": ["I miss them", "Life feels empty", "I want to cry"],
    "angry": ["This is infuriating", "I'm so mad right now", "Why can't they listen?"],
    "relaxed": ["Just chilling", "Feeling peaceful", "I’m enjoying the calm"],
    "excited": ["Can’t wait!", "This is amazing", "So thrilled about this"]
}

texts = []
labels = []
for mood, examples in mood_data.items():
    texts.extend(examples)
    labels.extend([mood] * len(examples))

model = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(texts, labels)

class InputText(BaseModel):
    sentence: str
    platform: Literal['youtube', 'spotify', 'jiomusic'] = 'youtube'

def generate_link(mood: str, platform: str) -> str:
    query = f"{mood} music"
    if platform == 'youtube':
        return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    elif platform == 'spotify':
        return f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
    elif platform == 'jiomusic':
        return f"https://www.jiosaavn.com/search/{query.replace(' ', '%20')}"
    return ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_mood")
def predict_mood(input_text: InputText):
    mood = model.predict([input_text.sentence])[0]
    suggestion_url = generate_link(mood, input_text.platform)
    return {
        "predicted_mood": mood,
        "suggestion_link": suggestion_url
    }

@app.get("/")
def root():
    return {"message": "Mood Music API is running. Go to /docs to use the API."}
