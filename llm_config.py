from dotenv import load_dotenv
import os

load_dotenv()

def get_llm_config():
    return {
        "config_list": [{
            "model": os.getenv("GROQ_MODEL", "llama3-70b-8192"),
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
            "api_type": "openai"
        }],
        "temperature": 0.1,
        "timeout": 120,
        "cache_seed": None
    }