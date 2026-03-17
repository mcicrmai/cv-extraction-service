import os
import json
import re
import pymupdf4llm
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
GENAI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GENAI_KEY)

# Detect working model
def get_working_model():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            return m.name
    return 'models/gemini-1.5-flash'

model = genai.GenerativeModel(get_working_model())
app = FastAPI()

# 2. CORS MIDDLEWARE (Fixes the 405/CORS error for Zoho)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    try:
        resume_text = pymupdf4llm.to_markdown(temp_filename)
        
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
            {{ "School": "", "Qualification": "Diploma/Degree/etc", "Major": "", "Summary": "", "From": "", "To": "" }}
          ],
          "WorkExperience": [
            {{ "Company": "", "JobTitle": "", "Summary": [ {{ "Description": "" }} ], "LeavingReason": "", "From": "", "To": "" }}
          ],
          "Address": {{ "PostalCode": "", "Floor": "", "UnitNumber": "" }},
          "OtherInformation": ""
        }}

        Resume Text:
        {resume_text}
        """
        
        response = model.generate_content(prompt)
        clean_json_str = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(clean_json_str)
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.get("/")
def home():
    return {"status": "API is live. Use POST /extract to parse resumes."}
