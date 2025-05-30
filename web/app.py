from flask import Flask, render_template, request, jsonify, send_file, session, redirect
from flask_cors import CORS
import sys
import os
from pathlib import Path
from groq import Groq
from werkzeug.utils import secure_filename
import time
import json
from datetime import datetime

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
from agents.psychiatrist_agent import PsychiatristAgent
from agents.context_analyzer_agent import ContextAnalyzerAgent
from engine.chat_loop import run_chat_session
import os
from dotenv import load_dotenv
import queue
import threading

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # Required for session

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
context_analyzer = ContextAnalyzerAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
planner_agent = ResponsePlannerAgent(context_analyzer=context_analyzer)
responder_agent = LLMResponderAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
sanitizer_agent = SanitizerAgent(known_vocabulary=profile.get("known_vocabulary", []), model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
psychiatrist_agent = PsychiatristAgent(model="llama3-70b-8192", api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

# Create a message queue for communication between threads
message_queue = queue.Queue()

# Add this near the top of the file with other configurations
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_chat_message(message, chat_history):
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
                psychiatrist_agent,
                save_profile,
                input_message=message,
                response_queue=response_queue,
                chat_history=chat_history
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
def index():
    return render_template('index.html')

@app.route('/care_taker')
def care_taker():
    return render_template('care_taker.html')

@app.route('/setup_profile', methods=['POST'])
def setup_profile():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Extract profile data
        profile_data = {
            'name': data.get('name'),
            'mental_age': data.get('mental_age'),
            'known_vocabulary': data.get('known_vocabulary', '').split(',') if data.get('known_vocabulary') else [],
            'trigger_words': data.get('trigger_words', '').split(',') if data.get('trigger_words') else [],
            'preferred_topics': data.get('preferred_topics', '').split(',') if data.get('preferred_topics') else []
        }
        
        # Save profile to JSONL file
        profile_path = os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl")
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f)
            f.write('\n')
        
        # Process core memories
        memories = []
        memory_texts = data.get('memory_text[]', [])
        tags = data.get('tags[]', [])
        vocabulary = data.get('vocabulary[]', [])
        emotions = data.get('emotion[]', [])
        types = data.get('type[]', [])
        importance_scores = data.get('memory_importance[]', [])
        
        # Convert string lists to actual lists if needed
        if isinstance(memory_texts, str):
            memory_texts = [memory_texts]
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(vocabulary, str):
            vocabulary = [vocabulary]
        if isinstance(emotions, str):
            emotions = [emotions]
        if isinstance(types, str):
            types = [types]
        if isinstance(importance_scores, str):
            importance_scores = [importance_scores]
        
        # Create memory objects
        for i in range(len(memory_texts)):
            memory = {
                'id': f'mem_{i+1:03d}',
                'text': memory_texts[i],
                'tags': tags[i].split(',') if tags[i] else [],
                'vocabulary': vocabulary[i].split(',') if vocabulary[i] else [],
                'emotion': emotions[i].split(',')[0] if emotions[i] else 'neutral',
                'memory_type': types[i].split(',')[0] if types[i] else 'experience',
                'importance_score': float(importance_scores[i]) if importance_scores[i] else 0.8,
                'last_used': datetime.now().strftime('%Y-%m-%d'),
                'used_count': 1
            }
            memories.append(memory)
        
        # Save memories to JSONL file
        memories_path = os.path.join(PROJECT_ROOT, "data", "core_memories.jsonl")
        with open(memories_path, 'w') as f:
            for memory in memories:
                json.dump(memory, f)
                f.write('\n')
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in setup_profile: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat')
def chat_page():
    # Initialize chat history if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Process the message using our chat system
        response = process_chat_message(user_message, chat_history)
        
        # Update chat history
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Assistant: {response}")
        session['chat_history'] = chat_history[-12:]  # Keep last 12 messages
        
        # Generate speech for the response
        client = Groq(api_key=groq_api_key)
        speech_file_path = os.path.join(PROJECT_ROOT, "web", "static", "speech.wav")
        
        # Ensure the speech file directory exists
        os.makedirs(os.path.dirname(speech_file_path), exist_ok=True)
        
        try:
            print(f"Generating speech for response: {response}")
            # Clean up any existing speech file
            if os.path.exists(speech_file_path):
                try:
                    os.remove(speech_file_path)
                    print("Removed existing speech file")
                except Exception as e:
                    print(f"Error removing existing speech file: {str(e)}")
            
            # Generate new speech file
            speech_response = client.audio.speech.create(
                model="playai-tts",
                voice="Jennifer-PlayAI",
                response_format="wav",
                input=response
            )
            
            # Use write_to_file method to save the audio
            speech_response.write_to_file(speech_file_path)
            print(f"Speech file saved to: {speech_file_path}")
            
            # Verify the file was created and is accessible
            if not os.path.exists(speech_file_path):
                raise Exception("Speech file was not created successfully")
                
            # Verify file permissions
            if not os.access(speech_file_path, os.R_OK):
                raise Exception("Speech file is not readable")
                
            return jsonify({
                'response': response,
                'status': 'success',
                'hasAudio': True
            })
        except Exception as e:
            print(f"Speech generation error: {str(e)}")
            # Return success with no audio if speech generation fails
            return jsonify({
                'response': response,
                'status': 'success',
                'hasAudio': False,
                'error': str(e)
            })
            
    except Exception as e:
        print(f"Error in chat session: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/speech')
def get_speech():
    speech_file_path = Path(__file__).parent / "static" / "speech.wav"
    print(f"Serving speech file from: {speech_file_path}")
    
    # Ensure the speech file exists and is accessible
    if not os.path.exists(speech_file_path):
        print("Speech file not found!")
        return jsonify({'error': 'Speech file not found'}), 404
        
    try:
        # Verify the file is readable
        if not os.access(speech_file_path, os.R_OK):
            print("Speech file is not readable!")
            return jsonify({'error': 'Speech file is not readable'}), 403
            
        # Read a small portion to verify
        with open(speech_file_path, 'rb') as f:
            # Just read a small portion to verify
            f.read(1024)
            
        return send_file(speech_file_path, mimetype='audio/wav')
    except Exception as e:
        print(f"Error serving speech file: {str(e)}")
        return jsonify({'error': 'Error accessing speech file'}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    print("Received transcription request")
    if 'audio' not in request.files:
        print("No audio file in request")
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Create a temporary file with a unique name
        temp_filename = f"temp_audio_{int(time.time())}.webm"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        print(f"Saving audio to: {filepath}")
        audio_file.save(filepath)
        
        # Transcribe the audio using Groq
        print("Starting transcription with Groq")
        with open(filepath, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                language="en",
                response_format="verbose_json",
            )
        
        print(f"Transcription result: {transcription.text}")
        
        # Clean up the temporary file
        os.remove(filepath)
        print("Temporary file cleaned up")
        
        return jsonify({'text': transcription.text})
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        # Clean up the temporary file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 