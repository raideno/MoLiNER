import os
import logging

from huggingface_hub import login

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    logger.info(f"[.env]: {load_dotenv()}")
except ImportError:
    def load_dotenv():
        return False
    logger.error("[.env]: dotenv not available")
    
from src.constants import (
    HUGGING_FACE_TOKEN
)

def login_to_huggingface(token: str=HUGGING_FACE_TOKEN):
    login(token=token)
