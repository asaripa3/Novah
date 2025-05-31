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
    # Load profile and memories data
    profile = load_profile(os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl"))
    memories = load_core_memories(os.path.join(PROJECT_ROOT, "data", "core_memories.jsonl"))
    return render_template('care_taker.html', profile=profile, memories=memories)

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
            # Clean up any existing speech file
            if os.path.exists(speech_file_path):
                try:
                    os.remove(speech_file_path)
                except Exception as e:
                    pass
            
            # Generate new speech file
            speech_response = client.audio.speech.create(
                model="playai-tts",
                voice="Jennifer-PlayAI",
                response_format="wav",
                input=response
            )
            
            # Use write_to_file method to save the audio
            speech_response.write_to_file(speech_file_path)
            
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
            # Return success with no audio if speech generation fails
            return jsonify({
                'response': response,
                'status': 'success',
                'hasAudio': False,
                'error': str(e)
            })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/speech')
def get_speech():
    speech_file_path = Path(__file__).parent / "static" / "speech.wav"
    
    # Ensure the speech file exists and is accessible
    if not os.path.exists(speech_file_path):
        return jsonify({'error': 'Speech file not found'}), 404
        
    try:
        # Verify the file is readable
        if not os.access(speech_file_path, os.R_OK):
            return jsonify({'error': 'Speech file is not readable'}), 403
            
        # Read a small portion to verify
        with open(speech_file_path, 'rb') as f:
            # Just read a small portion to verify
            f.read(1024)
            
        return send_file(speech_file_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': 'Error accessing speech file'}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Create a temporary file with a unique name
        temp_filename = f"temp_audio_{int(time.time())}.webm"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        audio_file.save(filepath)
        
        # Transcribe the audio using Groq
        with open(filepath, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                language="en",
                response_format="verbose_json",
            )
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({'text': transcription.text})
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_profile', methods=['POST'])
def update_profile():
    try:
        data = request.json
        profile_path = os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl")
        memories_path = os.path.join(PROJECT_ROOT, "data", "core_memories.jsonl")
        
        # Update profile
        with open(profile_path, 'w') as f:
            f.write(json.dumps(data['profile'], ensure_ascii=False) + '\n')
            
        # Update memories
        with open(memories_path, 'w') as f:
            for memory in data['memories']:
                f.write(json.dumps(memory, ensure_ascii=False) + '\n')
                
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/append_memory', methods=['POST'])
def append_memory():
    try:
        memory = request.json
        memories_path = os.path.join(PROJECT_ROOT, "data", "core_memories.jsonl")
        
        # Append the new memory to the file
        with open(memories_path, 'a+') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(f.tell() - 1)
                last_char = f.read(1)
                if last_char != '\n':
                    f.write('\n')
            f.write(json.dumps(memory, ensure_ascii=False) + '\n')
        
        # Reload all memories to ensure consistency
        memories = load_core_memories(memories_path)
                
        return jsonify({
            'status': 'success',
            'memories': memories
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/delete_memory', methods=['POST'])
def delete_memory():
    try:
        memory_id = request.json.get('id')
        memories_path = os.path.join(PROJECT_ROOT, "data", "core_memories.jsonl")
        profile_path = os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl")

        # Load all memories
        memories = load_core_memories(memories_path)
        # Remove the memory with the given id
        memories = [m for m in memories if m.get('id') != memory_id]
        # Write back to file
        with open(memories_path, 'w') as f:
            for memory in memories:
                f.write(json.dumps(memory, ensure_ascii=False) + '\n')

        # Load and update profile
        profile = load_profile(profile_path)
        # Recompute known_vocabulary, trigger_words, preferred_topics from remaining memories
        all_vocab = set()
        all_triggers = set()
        all_topics = set()
        for m in memories:
            all_vocab.update(m.get('vocabulary', []))
            all_triggers.update(m.get('tags', []))
            if m.get('memory_type'):
                all_topics.add(m.get('memory_type'))

        profile['known_vocabulary'] = sorted(list(all_vocab))
        profile['trigger_words'] = sorted(list(all_triggers))
        profile['preferred_topics'] = sorted(list(all_topics))

        # Save updated profile
        with open(profile_path, 'w') as f:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')

        return jsonify({'status': 'success', 'memories': memories, 'profile': profile})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/delete_profile_word', methods=['POST'])
def delete_profile_word():
    try:
        word = request.json.get('word')
        type_ = request.json.get('type')
        profile_path = os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl")
        profile = load_profile(profile_path)
        if type_ == 'vocabulary':
            profile['known_vocabulary'] = [w for w in profile.get('known_vocabulary', []) if w != word]
        elif type_ == 'trigger':
            profile['trigger_words'] = [w for w in profile.get('trigger_words', []) if w != word]
        elif type_ == 'topic':
            profile['preferred_topics'] = [w for w in profile.get('preferred_topics', []) if w != word]
        with open(profile_path, 'w') as f:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')
        return jsonify({'status': 'success', 'profile': profile})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/add_profile_word', methods=['POST'])
def add_profile_word():
    try:
        word = request.json.get('word')
        type_ = request.json.get('type')
        profile_path = os.path.join(PROJECT_ROOT, "data", "yahya_profile.jsonl")
        profile = load_profile(profile_path)
        if type_ == 'vocabulary':
            if word not in profile.get('known_vocabulary', []):
                profile['known_vocabulary'].append(word)
        elif type_ == 'trigger':
            if word not in profile.get('trigger_words', []):
                profile['trigger_words'].append(word)
        elif type_ == 'topic':
            if word not in profile.get('preferred_topics', []):
                profile['preferred_topics'].append(word)
        with open(profile_path, 'w') as f:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')
        return jsonify({'status': 'success', 'profile': profile})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)