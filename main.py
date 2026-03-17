import os
import json
import re
import pymupdf4llm
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. Setup & AI Configuration
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

# Auto-detect the best working model for your region
def get_working_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                return m.name
    except:
        pass
    return 'gemini-1.5-flash'

model = genai.GenerativeModel(get_working_model())
app = FastAPI()

# 2. CORS MIDDLEWARE - This fixes the 405 error in Zoho Widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Zoho domains to talk to Railway
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # OPTIONS is required for Zoho pre-flight
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    # Save the uploaded PDF temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    try:
        # Convert PDF to Markdown text
        resume_text = pymupdf4llm.to_markdown(temp_filename)

        # 3. Your Specific Singapore Recruitment Schema
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

        # 4. Generate AI response and clean JSON
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json|```', '', response.text).strip()
        
        return json.loads(clean_json)

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Always delete the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/")
def home():
    return {"status": "Online", "endpoint": "/extract", "method": "POST"}
