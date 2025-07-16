import os
import typing
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
    __HUGGING_FACE_TOKEN_ENV_VAR
)

def login_to_huggingface(token: typing.Optional[str] = None):
    token = token or os.getenv(__HUGGING_FACE_TOKEN_ENV_VAR)
    login(token=token)
