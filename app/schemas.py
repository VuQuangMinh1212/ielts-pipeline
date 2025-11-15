import uuid
import time
import os
from .db import SessionLocal, AudioRecord
from .genai_client import upload_file, transcribe, quality_check, ielts_score, upload_transcript_text
from .config import settings


# Simple file-based queue for demo (production: use Redis, RabbitMQ, or cloud queue)
QUEUE_DIR = '/data/queue'
os.makedirs(QUEUE_DIR, exist_ok=True)


def enqueue(local_audio_path, user_id):
task_id = str(uuid.uuid4())
payload = {"id": task_id, "path": local_audio_path, "user_id": user_id}
fname = os.path.join(QUEUE_DIR, task_id + '.json')
import json
with open(fname, 'w') as f:
json.dump(payload, f)
return task_id


def process_once():
files = [f for f in os.listdir(QUEUE_DIR) if f.endswith('.json')]
for fn in files:
try:
fp = os.path.join(QUEUE_DIR, fn)
import json
with open(fp) as f:
payload = json.load(f)
task_id = payload['id']
local_path = payload['path']
user_id = payload['user_id']


# 1. upload audio to genai
uploaded = upload_file(local_path, metadata={"user_id": user_id, "task_id": task_id})
file_id = uploaded.name


# 2. transcribe
t = transcribe(file_id)
transcript_text = t.get('transcript', '')
segments = t.get('segments', [])


# 3. quality check
q = quality_check(file_id)


# 4. ielts scoring
s = ielts_score(file_id)


# 5. upload transcript for File Search
fs_id = upload_transcript_text(transcript_text, metadata={"user_id": user_id, "task_id": task_id})


# 6. persist to DB
db = SessionLocal()
rec = AudioRecord(
id=task_id,
user_id=user_id,
audio_url=local_path,
transcript_text=transcript_text,
transcript_file_id=fs_id,
quality=q,
ielts=s,
segments=segments
)
db.add(rec)
db.commit()
db.close()


# remove queue file
os.remove(fp)
print(f"Processed {task_id}")
except Exception as e:
print("Error processing", fn, e)


if __name__ == '__main__':
while True:
process_once()
time.sleep(settings.WORKER_POLL_INTERVAL)