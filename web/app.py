from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.profile_loader import load_profile, save_profile
from utils.memory_loader import load_core_memories
from agents.memory_retrieval_agent import MemoryRetrievalAgent
from agents.query_parser import QueryParserAgent
from agents.context_filter_agent import ContextFilterAgent
from agents.response_planner_agent import ResponsePlannerAgent
from agents.llm_responder_agent import LLMResponderAgent
from agents.sanitizer_agent import SanitizerAgent
from engine.chat_loop import run_chat_session
import os
from dotenv import load_dotenv
import queue
import threading

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize agents with correct file paths
profile = load_profile(os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl"))

query_agent = QueryParserAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
memory_agent = MemoryRetrievalAgent(memory_file=os.path.join(PROJECT_ROOT, "data", "core_memories.jsonl"))
context_agent = ContextFilterAgent(known_vocabulary=profile.get("known_vocabulary", []))
planner_agent = ResponsePlannerAgent()
responder_agent = LLMResponderAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
sanitizer_agent = SanitizerAgent(known_vocabulary=profile.get("known_vocabulary", []), model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

# Create a message queue for communication between threads
message_queue = queue.Queue()

def process_chat_message(message):
    """Process a single chat message and return the response"""
    try:
        # Create a temporary queue for this message
        response_queue = queue.Queue()
        
        # Start the chat session in a separate thread
        def run_chat():
            run_chat_session(
                profile,
                query_agent,
                memory_agent,
                context_agent,
                planner_agent,
                responder_agent,
                sanitizer_agent,
                save_profile,
                input_message=message,
                response_queue=response_queue
            )
        
        # Start the chat thread
        chat_thread = threading.Thread(target=run_chat)
        chat_thread.start()
        
        # Wait for the response
        response = response_queue.get(timeout=30)  # 30 second timeout
        return response
    except Exception as e:
        return f"Error processing message: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Process the message using our chat system
        response = process_chat_message(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 