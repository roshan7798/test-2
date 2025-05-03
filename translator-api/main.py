from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import SeamlessM4Tv2ForTextToSpeech, AutoProcessor
import torchaudio, torch, base64, io

app = FastAPI()

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SeamlessM4Tv2ForTextToSpeech.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

class TranslationInput(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.post("/translate")
async def translate(input: TranslationInput):
    src_lang = "pes" if input.src_lang == "fa" else input.src_lang
    tgt_lang = "pes" if input.tgt_lang == "fa" else input.tgt_lang

    text_inputs = processor(text=input.text, src_lang=src_lang, return_tensors="pt").to(device)
    output = model.generate(**text_inputs, tgt_lang=tgt_lang, speaker_id=0, text_num_beams=4,
                            speech_do_sample=True, speech_temperature=0.6)

    audio = output[0].cpu().numpy().squeeze()
    text_tokens = output[2]
    translated_text = processor.decode(text_tokens.tolist()[0], skip_special_tokens=True)

    buffer = io.BytesIO()
    torchaudio.save(buffer, torch.from_numpy(audio).unsqueeze(0), 16000, format="wav")
    buffer.seek(0)

    audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return JSONResponse(content={
        "audio_base64": audio_b64,
        "translated_text": translated_text,
        "sample_rate": 16000
    })
