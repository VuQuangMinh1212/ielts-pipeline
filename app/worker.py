import os
import uuid
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from .worker import enqueue
from .db import init_db, SessionLocal, AudioRecord
from .schemas import UploadResponse, StatusResponse
from .config import settings


app = FastAPI(title="IELTS Pipeline")
init_db()


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile, background_tasks: BackgroundTasks, user_id: str):
# save file into /data/uploads
uploads_dir = '/data/uploads'
os.makedirs(uploads_dir, exist_ok=True)
file_id = str(uuid.uuid4())
local_path = os.path.join(uploads_dir, file_id + '_' + file.filename)
with open(local_path, 'wb') as f:
f.write(await file.read())


task_id = enqueue(local_path, user_id)
return UploadResponse(id=task_id, message='Uploaded and queued for processing')


@app.get('/status/{task_id}', response_model=StatusResponse)
def status(task_id: str):
db = SessionLocal()
rec = db.query(AudioRecord).filter(AudioRecord.id == task_id).first()
if not rec:
return JSONResponse(status_code=404, content={'detail': 'not found'})
return StatusResponse(id=rec.id, status='processed', quality=rec.quality, ielts=rec.ielts)