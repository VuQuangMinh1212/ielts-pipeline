from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
id: str
message: str


class StatusResponse(BaseModel):
id: str
status: str
quality: Optional[dict]
ielts: Optional[dict]

### `app/genai_client.py`
```python
import os
import json
from google import genai


from .config import settings


client = genai.Client(api_key=settings.GENAI_API_KEY)


# Upload file helper
def upload_file(path: str, metadata: dict = None):
# returns file object with `.name` or `.id` depending on SDK
return client.files.upload(path=path, metadata=metadata or {})


# Transcribe and return JSON {transcript, segments}
def transcribe(file_id: str):
prompt = '''You are an expert transcriber. Produce a JSON object:
{ "transcript": "...", "segments": [{"start":"00:00.00","end":"00:02.50","text":"..."}, ...] }
Return ONLY valid JSON.
'''
resp = client.models.generate_content(
model="gemini-1.5-pro",
contents=prompt,
file_ids=[file_id]
)
# Attempt to parse JSON
try:
return json.loads(resp.text)
except Exception:
return {"transcript": resp.text, "segments": []}


# Quality check
def quality_check(file_id: str):
prompt = '''You are an audio QA. Return JSON:
{ "quality_score": int, "noise_level":"low|moderate|high", "clarity":"clear|muffled|incoherent", "speech_rate_wpm": float, "recommendation":"ok|needs_preprocessing|discard", "notes":"..." }
Return ONLY JSON.
'''
resp = client.models.generate_content(
model="gemini-1.5-pro",
contents=prompt,
file_ids=[file_id]
)
try:
return json.loads(resp.text)
except Exception:
return {"quality_score": 5, "notes": resp.text}


# IELTS scoring
def ielts_score(file_id: str):
prompt = '''You are an IELTS certified examiner. Score the audio and return JSON:
{ "fluency":0-9, "lexical":0-9, "grammar":0-9, "pronunciation":0-9, "overall":0-9, "feedback":"..." }
Return ONLY JSON.
'''
resp = client.models.generate_content(
model="gemini-1.5-pro",
contents=prompt,
file_ids=[file_id]
)
try:
return json.loads(resp.text)
except Exception:
return {"overall": None, "feedback": resp.text}


# Upload transcript file for File Search
def upload_transcript_text(transcript_text: str, metadata: dict = None):
import tempfile
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
f.write(transcript_text)
tmp_path = f.name
uploaded = client.files.upload(path=tmp_path, metadata=metadata or {})
return uploaded.name