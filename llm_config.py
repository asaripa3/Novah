from dotenv import load_dotenv
import os

# Load environment from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=env_path)

def get_llm_config():
    return {
        "config_list": [{
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
            "api_type": "openai"
        }],
        "temperature": 0.1,
        "timeout": 120,
        "cache_seed": None
    }
