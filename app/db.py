from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from .config import settings


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AudioRecord(Base):
__tablename__ = 'audio_records'
id = Column(String, primary_key=True, index=True)
user_id = Column(String, index=True)
audio_url = Column(String)
transcript_text = Column(Text)
transcript_file_id = Column(String)
quality = Column(JSON)
ielts = Column(JSON)
segments = Column(JSON)
created_at = Column(DateTime, default=datetime.datetime.utcnow)
updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


def init_db():
Base.metadata.create_all(bind=engine)