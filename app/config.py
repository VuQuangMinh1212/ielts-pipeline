from pydantic import BaseSettings


class Settings(BaseSettings):
GENAI_API_KEY: str
DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@db:5432/ielts"
STORAGE_BASE_URL: str = ""
WORKER_POLL_INTERVAL: int = 2


class Config:
env_file = ".env"


settings = Settings()