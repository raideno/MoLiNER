import os

from huggingface_hub import login

try:
    from dotenv import load_dotenv
    print("[.env]:", load_dotenv())
except ImportError:
    def load_dotenv():
        return False
    print("[.env]: dotenv not available")
    
from src.constants import HUGGING_FACE_TOKEN

def login_to_huggingface(token: str=HUGGING_FACE_TOKEN):
    login(token=token)
