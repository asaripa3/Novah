from utils.profile_loader import load_profile, save_profile
from utils.memory_loader import load_core_memories
from agents.memory_retrieval_agent import MemoryRetrievalAgent
from agents.query_parser import QueryParserAgent
from agents.context_filter_agent import ContextFilterAgent
from agents.response_planner_agent import ResponsePlannerAgent
from agents.llm_responder_agent import LLMResponderAgent
from agents.sanitizer_agent import SanitizerAgent
from agents.psychiatrist_agent import PsychiatristAgent
from engine.chat_loop import run_chat_session
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

profile = load_profile("data/yahya_profile.jsonl")

query_agent = QueryParserAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
memory_agent = MemoryRetrievalAgent(memory_file="data/core_memories.jsonl")
context_agent = ContextFilterAgent(known_vocabulary=profile.get("known_vocabulary", []))
planner_agent = ResponsePlannerAgent()
responder_agent = LLMResponderAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
sanitizer_agent = SanitizerAgent(known_vocabulary=profile.get("known_vocabulary", []), model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
psychiatrist_agent = PsychiatristAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

run_chat_session(profile, query_agent, memory_agent, context_agent, planner_agent, responder_agent, sanitizer_agent, psychiatrist_agent, save_profile)