# NovahSpeaks

NovahSpeaks is an innovative Emotional Intelligence (EQ) agent designed specifically for neurodiverse individuals. It provides personalized, context-aware responses by leveraging core memories and maintaining a consistent, supportive communication style.

## 🌟 Features

- **Personalized Communication**: Tailored responses based on individual's core memories and experiences
- **Context-Aware**: Maintains conversation context and emotional awareness
- **Simplified Language**: Uses appropriate vocabulary and clear explanations
- **Emotional Support**: Maintains a consistent, supportive tone throughout interactions
- **Memory Integration**: Incorporates relevant past experiences into responses

## 🏗️ Architecture

The system is built on a multi-agent architecture that works together to provide personalized responses:

### Core Components

1. **QueryParserAgent**
   - Analyzes user questions
   - Extracts intent, emotions, and key topics
   - Identifies the communication context

2. **MemoryRetrievalAgent**
   - Manages core memories database
   - Uses vector similarity search for relevant memories
   - Integrates with FAISS/Pinecone for efficient retrieval

3. **ContextFilterAgent**
   - Filters memories based on:
     - Emotional alignment
     - Vocabulary fit
     - Recency and relevance
   - Ensures appropriate cognitive load

4. **ResponsePlannerAgent**
   - Crafts structured prompts
   - Integrates selected memories with current context
   - Maintains consistent communication style

5. **LLMResponderAgent**
   - Generates responses using LLM
   - Maintains tone and style consistency
   - Ensures appropriate length and complexity

6. **SanitizerAgent**
   - Validates response safety
   - Ensures vocabulary compliance
   - Maintains consistent tone

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Groq API key
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NovahSpeaks.git
cd NovahSpeaks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

4. Prepare your data:
- Add core memories to `data/core_memories.jsonl`
- Configure user profile in `data/yahya_profile.jsonl`

### Usage

Run the main application:
```bash
python main.py
```

## 📁 Project Structure

```
NovahSpeaks/
├── agents/           # Agent implementations
├── data/            # Core memories and profiles
├── engine/          # Core engine components
├── utils/           # Utility functions
├── main.py          # Application entry point
├── llm_config.py    # LLM configuration
└── README.md        # This file
```

