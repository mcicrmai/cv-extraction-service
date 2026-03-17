import os
import json
import re
import pymupdf4llm
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Detect working model
def get_working_model():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            return m.name
    return 'models/gemini-1.5-flash'

WORKING_MODEL = get_working_model()
model = genai.GenerativeModel(WORKING_MODEL)

app = FastAPI()

# --- 1. ADDED CORS (Allows Zoho Widget to talk to Localhost) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    try:
        resume_text = pymupdf4llm.to_markdown(temp_filename)
        
        # --- 2. THE DETAILED ZOHO PROMPT ---
        prompt = f"""
        Parse this resume for recruitment purposes. Extract data and return ONLY valid JSON in this exact format:
        {{
          "Name": "",
          "DateOfBirth": "",
          "Age": "",
          "Gender": "Male or Female",
          "Race": "Chinese, Malay, Indian, Eurasian, Punjabi, Filipino, Caucasian, Burmese, Thai, Bangladeshi, Sri Lankan, Japanese, Korean, or Others",
          "Nationality": "Use country name except Singaporean, Malaysian",
          "Residency": "Citizen, Permanent Resident, Work Permit, S Pass, Employment Pass, Long Term Visit Pass, Student Pass, Dependant Pass",
          "NoticePeriod": "",
          "Mobile": "+65 XXXX XXXX",
          "Email": "",
          "ProfileSummary": "",
          "Languages": "",
          "Skills": "",
          "LastDrawnSalary": "",
          "ExpectedSalary": "",
          "NearestMRTStation": "",
          "Education": [
            {{
              "School": "",
              "Qualification": "O Level, A Level, Nitec, Higher Nitec, Diploma, Bachelor's Degree, Master's Degree, PhD, Professional Certificate, Others",
              "Major": "",
              "Summary": "",
              "From": "",
              "To": ""
            }}
          ],
          "WorkExperience": [
            {{
              "Company": "",
              "JobTitle": "",
              "Summary": [ {{ "Description": "" }} ],
              "LeavingReason": "",
              "From": "",
              "To": ""
            }}
          ],
          "Address": {{ "PostalCode": "", "Floor": "", "UnitNumber": "" }},
          "OtherInformation": ""
        }}

        Resume Text:
        {resume_text}
        """
        
        response = model.generate_content(prompt)
        # Remove any markdown code blocks from AI response
        clean_json_str = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(clean_json_str)
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.get("/")
def home():
    return {"status": "Local API is live", "model": WORKING_MODEL}
