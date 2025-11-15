CREATE TABLE IF NOT EXISTS audio_records (
id TEXT PRIMARY KEY,
user_id TEXT,
audio_url TEXT,
transcript_text TEXT,
transcript_file_id TEXT,
quality JSON,
ielts JSON,
segments JSON,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);