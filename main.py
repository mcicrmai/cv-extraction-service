import os
import json
import re
import pymupdf4llm
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- NEW: AUTO-DETECT WORKING MODEL ---
def get_working_model():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            return m.name
    return 'models/gemini-1.5-flash' # Fallback

WORKING_MODEL = get_working_model()
print(f"Using confirmed working model: {WORKING_MODEL}")

model = genai.GenerativeModel(WORKING_MODEL)
app = FastAPI()

@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    try:
        resume_text = pymupdf4llm.to_markdown(temp_filename)
        prompt = f"Extract resume data into JSON: {resume_text}"
        
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
