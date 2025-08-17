import os
from pydantic_settings import BaseSettings
from pinecone import Pinecone, ServerlessSpec
import getpass
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="proto.marshal.rules.enums")

# Configuration management
class Settings(BaseSettings):
    QB_CLIENT_ID: str
    QB_CLIENT_SECRET: str
    QB_REFRESH_TOKEN: str
    QB_REALM_ID: str
    QB_BASE_URL: str = "https://sandbox-quickbooks.api.intuit.com"
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "langchain-pinecone-rag"
    UPLOAD_DIR: str = "Uploads"
    GOOGLE_APPLICATION_CREDENTIALS: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load environment variables and validate
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
settings = Settings()

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = settings.PINECONE_INDEX

# Create Pinecone index if it doesn't exist
if index_name not in [index['name'] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )