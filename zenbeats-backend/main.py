from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from uuid import uuid4
from PIL import Image
import os
import torch
from diffusers import StableDiffusionPipeline

# === FastAPI App ===
app = FastAPI()

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Image Output Directory ===
IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# === Load Stable Diffusion Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None  # optionally disable NSFW checker
).to(device)

# === Emotion to Prompt Mapping ===
emotion_prompts = {
    "happy": "beautiful anime-style forest on a sunny day, vibrant colors, joy",
    "sad": "lonely anime figure by a rainy window, blue tones, emotional, melancholy",
    "angry": "peaceful zen garden at sunset, soft light, calming anime scene",
    "relaxed": "anime character meditating in a forest, tranquil, soft pastels",
    "excited": "anime-style fireworks over a cityscape, vibrant night, joyful crowd"
}

# === Request Model ===
class ArtRequest(BaseModel):
    emotion: Literal["happy", "sad", "angry", "relaxed", "excited"]

# === Generate Image from Prompt ===
def generate_image(prompt: str) -> str:
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(IMAGE_DIR, filename)
    image.save(filepath)
    return filepath

# === Root Endpoint ===
@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion-Based Art Generator API. Use /generate_art to create images."}

# === Image Generation Endpoint ===
@app.post("/generate_art")
def generate_art(request: ArtRequest):
    prompt = emotion_prompts.get(request.emotion, "anime fantasy landscape")
    try:
        image_path = generate_image(prompt)
        return {
            "emotion": request.emotion,
            "prompt_used": prompt,
            "image_url": f"/download/{os.path.basename(image_path)}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Download Image Endpoint ===
@app.get("/download/{filename}")
def download_image(filename: str):
    filepath = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Image not found."})
    return FileResponse(filepath, media_type="image/png", filename=filename)

# === 404 Custom Handler ===
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "The requested resource was not found. Please check your URL."},
    )
