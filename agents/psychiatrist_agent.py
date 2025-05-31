import json
from datetime import datetime
from typing import Dict, List, Optional
import openai
from utils.text_utils import normalize_list
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from pathlib import Path

class PsychiatristAgent:
    def __init__(self, model="llama3-70b-8192", api_key=None, base_url=None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Use absolute paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.memory_file = os.path.join(self.base_dir, "data", "core_memories.jsonl")
        self.profile_file = os.path.join(self.base_dir, "data", "yahya_profile.jsonl")
        
        print("\n[Initializing Psychiatrist Agent]")
        print(f"Base directory: {self.base_dir}")
        print(f"Memory file path: {self.memory_file}")
        print(f"Profile file path: {self.profile_file}")
        
        # Create data directory if it doesn't exist
        data_dir = os.path.dirname(self.memory_file)
        if not os.path.exists(data_dir):
            print(f"Creating data directory: {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
        
        # Create empty memory file if it doesn't exist
        if not os.path.exists(self.memory_file):
            print(f"Creating new memory file: {self.memory_file}")
            with open(self.memory_file, 'w') as f:
                pass
        
        # Download required NLTK data
        print("\n[Initializing NLTK Data]")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading punkt tokenizer...")
            nltk.download('punkt')
            
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            print("Downloading POS tagger...")
            nltk.download('averaged_perceptron_tagger')
            
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            print("Downloading VADER lexicon...")
            nltk.download('vader_lexicon')
            
        print("Initializing sentiment analyzer...")
        self.sia = SentimentIntensityAnalyzer()
        print("NLTK initialization complete!\n")

        self.tool_instructions = {
            "emotion_analysis": {
                "description": "Analyze emotional content and context",
                "usage": "Use to understand emotional state and triggers",
                "output_format": "Returns emotional analysis with primary and secondary emotions"
            },
            "memory_evaluation": {
                "description": "Evaluate if content should be saved as memory",
                "usage": "Use to determine memory significance",
                "output_format": "Returns memory evaluation with significance score"
            },
            "vocabulary_analysis": {
                "description": "Analyze and categorize vocabulary",
                "usage": "Use to identify meaningful and emotional words",
                "output_format": "Returns categorized word lists"
            }
        }

    def _load_memories(self) -> List[Dict]:
        """Load memories from file with error handling."""
        try:
            if not os.path.exists(self.memory_file):
                print(f"Memory file not found: {self.memory_file}")
                return []
            
            # Try to repair the file if it exists
            self._repair_memory_file()
                
            memories = []
            with open(self.memory_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        memory = json.loads(line.strip())
                        # Ensure memory has all required fields
                        if "memory_type" not in memory:
                            if memory.get("emotion") in ["happy", "excited", "proud"]:
                                memory["memory_type"] = "achievement"
                            elif memory.get("emotion") in ["sad", "scared", "anxious"]:
                                memory["memory_type"] = "fear"
                            else:
                                memory["memory_type"] = "experience"
                        memories.append(memory)
                    except json.JSONDecodeError as e:
                        print(f"! Warning: Could not parse memory at line {line_num}: {str(e)}")
                        print(f"Line content: {line.strip()}")
                        continue
                        
            print(f"Loaded {len(memories)} memories from file")
            return memories
        except Exception as e:
            print(f"Error loading memories: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Memory file path: {self.memory_file}")
            return []

    def _save_memory(self, memory: Dict):
        """Save memory to file with immediate flushing and verification."""
        print("\n[Saving Memory]")
        print(f"Writing to file: {self.memory_file}")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Convert memory to JSON string with proper formatting
            memory_json = json.dumps(memory, ensure_ascii=False)
            
            # Add newline before writing if file is not empty
            with open(self.memory_file, 'a') as f:
                if os.path.getsize(self.memory_file) > 0:
                    f.write('\n')  # Add newline before new memory
                f.write(memory_json)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure OS writes to disk
            
            # Verify the write
            with open(self.memory_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print("! Warning: File is empty after write")
                    return
                    
                last_line = lines[-1].strip()
                try:
                    saved_memory = json.loads(last_line)
                    if saved_memory["id"] == memory["id"]:
                        print("✓ Memory successfully saved and verified")
                        print(f"Current memory count: {len(lines)}")
                    else:
                        print("! Warning: Memory verification failed")
                except json.JSONDecodeError as e:
                    print(f"! Warning: Could not verify memory - JSON decode error: {str(e)}")
                    print(f"Last line content: {last_line}")
        except Exception as e:
            print(f"! Error saving memory: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Memory file path: {self.memory_file}")
            print(f"Memory content: {json.dumps(memory, indent=2)}")
            raise

    def _repair_memory_file(self):
        """Repair corrupted memory file by reading and rewriting valid memories."""
        print("\n[Repairing Memory File]")
        try:
            if not os.path.exists(self.memory_file):
                print("Memory file not found, no repair needed")
                return
                
            # Read all lines and try to parse each as JSON
            valid_memories = []
            with open(self.memory_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        memory = json.loads(line)
                        # Add memory_type if missing
                        if "memory_type" not in memory:
                            # Infer memory type from emotion and content
                            if memory.get("emotion") in ["happy", "excited", "proud"]:
                                memory["memory_type"] = "achievement"
                            elif memory.get("emotion") in ["sad", "scared", "anxious"]:
                                memory["memory_type"] = "fear"
                            else:
                                memory["memory_type"] = "experience"
                        valid_memories.append(memory)
                    except json.JSONDecodeError:
                        print(f"! Skipping invalid memory line: {line[:100]}...")
                        continue
            
            # Write back only valid memories
            with open(self.memory_file, 'w') as f:
                for i, memory in enumerate(valid_memories):
                    if i > 0:
                        f.write('\n')
                    f.write(json.dumps(memory, ensure_ascii=False))
            
            print(f"✓ Memory file repaired. Valid memories: {len(valid_memories)}")
        except Exception as e:
            print(f"! Error repairing memory file: {str(e)}")

    def _load_profile(self) -> Dict:
        try:
            with open(self.profile_file, 'r') as f:
                return json.loads(f.read())
        except FileNotFoundError:
            return {"known_vocabulary": [], "trigger_words": [], "preferred_topics": []}

    def _save_profile(self, profile: Dict):
        os.makedirs(os.path.dirname(self.profile_file), exist_ok=True)
        with open(self.profile_file, 'w') as f:
            json.dump(profile, f)

    def _analyze_emotion(self, text: str) -> Dict:
        """Analyze emotional content of text."""
        print("\n[Emotion Analysis]")
        print(f"Analyzing text: {text}")
        
        # Thought process structure
        thought_process = f"""
        Thought Process:
        1. Text Analysis:
           - Input Text: {text}
           - Sentiment Scores: To be calculated
           - Emotional Cues: To be identified
        
        2. Emotion Detection:
           - Primary Emotion: To be determined
           - Secondary Emotions: To be identified
           - Emotional Intensity: To be measured
        
        3. Context Evaluation:
           - Triggering Content: To be assessed
           - Support Needs: To be evaluated
           - Emotional Impact: To be considered
        
        4. Final Integration:
           - Combine analyses
           - Structure output
           - Ensure completeness
        """

        # Tool usage instructions
        tool_instructions = """
        Available Tools:
        1. Sentiment Analysis Tool:
           - Purpose: Calculate sentiment scores
           - Usage: Analyze emotional tone
           - Output: Sentiment scores and emotional indicators

        2. Emotion Detection Tool:
           - Purpose: Identify specific emotions
           - Usage: Determine primary and secondary emotions
           - Output: List of emotions with intensities

        3. Context Analysis Tool:
           - Purpose: Evaluate emotional context
           - Usage: Assess triggering content and support needs
           - Output: Contextual analysis with recommendations
        """
        
        # Get sentiment scores
        sentiment_scores = self.sia.polarity_scores(text)
        print(f"Sentiment scores: {sentiment_scores}")
        
        # Use LLM for detailed emotion analysis
        prompt = (
            f"{thought_process}\n\n"
            f"{tool_instructions}\n\n"
            "Analyze the emotional content of this text from a neurodiverse child's perspective. "
            "Consider both explicit and implicit emotions. Return a JSON with:\n"
            "- primary_emotion: The main emotion (e.g., happy, anxious, curious, confused)\n"
            "- secondary_emotions: List of other emotions present\n"
            "- emotional_intensity: 0-1 scale\n"
            "- is_triggering: boolean indicating if content might be triggering\n"
            "- needs_support: boolean indicating if emotional support is needed\n\n"
            f"Text: {text}\n\n"
            "Return ONLY the JSON object, no other text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in child psychology and neurodiversity. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        try:
            content = response.choices[0].message.content.strip()
            # Clean up the response to ensure it's valid JSON
            content = content.replace('\n', ' ').replace('\r', '')
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            emotion_analysis = json.loads(content)
            print(f"Emotion analysis result: {json.dumps(emotion_analysis, indent=2)}")
            return emotion_analysis
        except json.JSONDecodeError as e:
            print(f"Error parsing emotion analysis JSON: {str(e)}")
            print(f"Raw response: {content}")
            return {
                "primary_emotion": "neutral",
                "secondary_emotions": [],
                "emotional_intensity": 0.5,
                "is_triggering": False,
                "needs_support": False
            }

    def _should_add_to_vocabulary(self, word: str) -> bool:
        """Determine if a word should be added to vocabulary."""
        # Check if word is too common
        if word.lower() in self.common_words:
            return False
            
        # Check if word is too short (less than 3 characters)
        if len(word) < 3:
            return False
            
        # Check if word is a number
        if word.isdigit():
            return False
            
        return True

    def _evaluate_words(self, words: List[str], context: str) -> Dict[str, List[str]]:
        """Use LLM to evaluate which words are meaningful for vocabulary."""
        print("\n[Word Evaluation]")
        print(f"Evaluating words: {words}")
        
        prompt = (
            "You are an expert in child psychology and language development. "
            "Analyze these words from a neurodiverse child's message and categorize them.\n\n"
            "Context: " + context + "\n\n"
            "Words to evaluate: " + ", ".join(words) + "\n\n"
            "Return a JSON with these categories:\n"
            "- meaningful_words: Words that are important for vocabulary building (nouns, key verbs, descriptive words)\n"
            "- common_words: Basic words that don't need to be in vocabulary (pronouns, articles, common verbs)\n"
            "- emotional_words: Words that carry emotional significance\n\n"
            "Consider:\n"
            "1. Is this word important for understanding the child's experiences?\n"
            "2. Is this a word that would be useful in future conversations?\n"
            "3. Does this word carry emotional or experiential significance?\n\n"
            "Return ONLY the JSON object, no other text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in child psychology and language development. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        try:
            content = response.choices[0].message.content.strip()
            # Clean up the response to ensure it's valid JSON
            content = content.replace('\n', ' ').replace('\r', '')
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            word_categories = json.loads(content)
            # Ensure all required fields are present
            if 'meaningful_words' not in word_categories:
                word_categories['meaningful_words'] = []
            if 'common_words' not in word_categories:
                word_categories['common_words'] = []
            if 'emotional_words' not in word_categories:
                word_categories['emotional_words'] = []
                
            print(f"Word evaluation result: {json.dumps(word_categories, indent=2)}")
            return word_categories
        except json.JSONDecodeError as e:
            print(f"Error parsing word evaluation JSON: {str(e)}")
            print(f"Raw response: {content}")
            return {
                "meaningful_words": [],
                "common_words": words,
                "emotional_words": []
            }

    def _evaluate_memory(self, text: str, emotion_analysis: Dict, keywords: List[str]) -> Dict:
        """Evaluate if content should be saved as a memory."""
        print("\n[Memory Evaluation]")
        print(f"Evaluating text: {text}")
        
        # Thought process structure
        thought_process = f"""
        Thought Process:
        1. Content Analysis:
           - Input Text: {text}
           - Emotional Context: {emotion_analysis}
           - Key Concepts: {keywords}
        
        2. Memory Significance:
           - Developmental Impact: To be assessed
           - Emotional Importance: To be evaluated
           - Future Relevance: To be considered
        
        3. Memory Categorization:
           - Memory Type: To be determined
           - Significance Score: To be calculated
           - Reasoning: To be provided
        
        4. Final Decision:
           - Save Decision: To be made
           - Memory Format: To be determined
           - Implementation: To be planned
        """

        # Tool usage instructions
        tool_instructions = """
        Available Tools:
        1. Significance Analysis Tool:
           - Purpose: Evaluate memory importance
           - Usage: Assess developmental and emotional impact
           - Output: Significance score and reasoning

        2. Memory Categorization Tool:
           - Purpose: Determine memory type
           - Usage: Classify memory based on content
           - Output: Memory type and characteristics

        3. Decision Support Tool:
           - Purpose: Guide memory saving decision
           - Usage: Evaluate save criteria
           - Output: Save decision with reasoning
        """

        prompt = (
            f"{thought_process}\n\n"
            f"{tool_instructions}\n\n"
            "Evaluate if this memory is worth saving and its significance.\n\n"
            "Return a JSON with:\n"
            "- should_save: boolean indicating if this memory is worth saving\n"
            "- significance_score: 0-1 scale of memory importance\n"
            "- reasoning: brief explanation of the decision\n"
            "- memory_type: type of memory (e.g., 'achievement', 'fear', 'milestone', 'preference', 'experience')\n\n"
            "Consider these strict criteria for saving a memory:\n"
            "1. Is this a significant experience or milestone for the child?\n"
            "2. Does it have high emotional or developmental importance?\n"
            "3. Would this memory be crucial for future interactions and understanding?\n"
            "4. Is it a core preference, fear, or achievement?\n"
            "5. Does it provide insight into the child's personality or needs?\n"
            "6. Is this memory unique and not redundant with existing memories?\n"
            "7. Does it represent a meaningful pattern or preference?\n\n"
            "DO NOT save:\n"
            "- Casual greetings or small talk\n"
            "- Simple questions or statements\n"
            "- Routine activities without emotional significance\n"
            "- Temporary states or passing thoughts\n"
            "- Redundant or similar memories\n"
            "- Generic or vague statements\n"
            "- Common preferences without deeper meaning\n\n"
            "A core memory should:\n"
            "- Reveal something important about the child's personality\n"
            "- Show a significant preference or fear\n"
            "- Represent a meaningful achievement or milestone\n"
            "- Provide insight into the child's emotional needs\n"
            "- Help understand the child's unique perspective\n\n"
            "Only save memories that are truly meaningful and will be valuable for future interactions.\n"
            "Return ONLY the JSON object, no other text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in child psychology and memory formation. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        try:
            content = response.choices[0].message.content.strip()
            # Clean up the response to ensure it's valid JSON
            content = content.replace('\n', ' ').replace('\r', '')
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            memory_evaluation = json.loads(content)
            print(f"Memory evaluation result: {json.dumps(memory_evaluation, indent=2)}")
            return memory_evaluation
        except json.JSONDecodeError as e:
            print(f"Error parsing memory evaluation JSON: {str(e)}")
            print(f"Raw response: {content}")
            return {
                "should_save": False,
                "significance_score": 0.0,
                "reasoning": "Error in evaluation",
                "memory_type": "unknown"
            }

    def _extract_keywords(self, text: str) -> List[str]:
        print("\n[Keyword Extraction]")
        print(f"Extracting keywords from: {text}")
        
        # Tokenize and get POS tags
        tokens = word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract nouns, verbs, and adjectives
        keywords = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ'))]
        keywords = normalize_list(keywords)
        
        # Get word evaluation from LLM
        word_categories = self._evaluate_words(keywords, text)
        
        print(f"All extracted keywords: {keywords}")
        print(f"Meaningful keywords: {word_categories['meaningful_words']}")
        print(f"Emotional words: {word_categories['emotional_words']}")
        
        # Combine meaningful and emotional words
        return list(set(word_categories['meaningful_words'] + word_categories['emotional_words']))

    def process_input(self, user_input: str, query_result: Dict) -> Dict:
        """Process user input and update profile/memories accordingly."""
        print("\n[Psychiatrist Agent] Processing new input")
        print(f"User input: {user_input}")
        print(f"Query result: {query_result}")

        # Thought process structure
        thought_process = f"""
        Thought Process:
        1. Input Analysis:
           - User Input: {user_input}
           - Query Context: {query_result}
           - Initial Assessment: To be performed
        
        2. Emotional Processing:
           - Emotion Analysis: To be conducted
           - Trigger Assessment: To be evaluated
           - Support Needs: To be determined
        
        3. Memory Processing:
           - Memory Evaluation: To be performed
           - Vocabulary Analysis: To be conducted
           - Profile Updates: To be planned
        
        4. Final Integration:
           - Combine analyses
           - Update profile
           - Save memories
        """

        # Tool usage instructions
        tool_instructions = """
        Available Tools:
        1. Emotion Analysis Tool:
           - Purpose: Understand emotional context
           - Usage: Analyze emotional content
           - Output: Emotional analysis with recommendations

        2. Memory Processing Tool:
           - Purpose: Evaluate and save memories
           - Usage: Process significant experiences
           - Output: Memory evaluations and updates

        3. Profile Update Tool:
           - Purpose: Maintain user profile
           - Usage: Update preferences and vocabulary
           - Output: Profile updates and changes
        """

        # Analyze emotion
        emotion_analysis = self._analyze_emotion(user_input)
        
        # Extract keywords
        keywords = self._extract_keywords(user_input)
        
        # Evaluate memory
        memory_evaluation = self._evaluate_memory(user_input, emotion_analysis, keywords)
        
        # Create memory entry if worth saving
        memory = None
        if memory_evaluation["should_save"]:
            # Format the memory text in a personal, first-person narrative style
            prompt = (
                f"{thought_process}\n\n"
                f"{tool_instructions}\n\n"
                "You are an expert in child psychology and memory formation. "
                "Rewrite this memory from Yahya's perspective, focusing on his feelings and experiences.\n\n"
                "Original text: " + user_input + "\n\n"
                "Guidelines:\n"
                "1. Write in first person (e.g., 'I felt', 'I remember')\n"
                "2. Focus on emotional experiences and feelings\n"
                "3. Include how Yahya felt at the time and how he feels now\n"
                "4. Use simple, clear language appropriate for a 10-year-old\n"
                "5. Emphasize personal reactions and emotional responses\n"
                "6. Keep it concise but meaningful\n"
                "7. Include both past and present feelings when relevant\n"
                "8. Use phrases like 'I felt', 'I remember', 'I wish', 'I understand'\n\n"
                "Example format:\n"
                "'I felt [emotion] when [situation]. I couldn't [express/understand] at the time, but now I [current understanding/feeling].'\n\n"
                "Return ONLY the rewritten memory text, no other text."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in child psychology and memory formation. Return only the rewritten memory text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            formatted_text = response.choices[0].message.content.strip()
            
            memory = {
                "id": f"mem_{len(self._load_memories()) + 1:03d}",
                "text": formatted_text,
                "tags": keywords,
                "emotion": emotion_analysis["primary_emotion"],
                "vocabulary": keywords,
                "last_used": datetime.now().strftime("%Y-%m-%d"),
                "used_count": 1,
                "importance_score": memory_evaluation["significance_score"],
                "memory_type": memory_evaluation["memory_type"]
            }
            
            print("\n[Memory Creation]")
            print(f"Created memory: {json.dumps(memory, indent=2)}")
            print(f"Reasoning: {memory_evaluation['reasoning']}")
            self._save_memory(memory)
        else:
            print("\n[Memory Creation]")
            print(f"Memory not saved: {memory_evaluation['reasoning']}")
        
        # Update profile
        profile = self._load_profile()
        
        # Update known vocabulary with meaningful words
        for word in keywords:
            if word not in profile["known_vocabulary"]:
                profile["known_vocabulary"].append(word)
                print(f"Added new word to vocabulary: {word}")
        
        # Update trigger words
        if emotion_analysis["is_triggering"]:
            for word in keywords:
                if word not in profile["trigger_words"]:
                    profile["trigger_words"].append(word)
                    print(f"Added new trigger word: {word}")
        
        # Update preferred topics
        if emotion_analysis["primary_emotion"] in ["happy", "excited", "curious"]:
            for word in keywords:
                if word not in profile["preferred_topics"]:
                    profile["preferred_topics"].append(word)
                    print(f"Added new preferred topic: {word}")
        
        # Save updated profile
        self._save_profile(profile)
        
        print("\n[Profile Update Summary]")
        print(f"Known vocabulary size: {len(profile['known_vocabulary'])}")
        print(f"Trigger words size: {len(profile['trigger_words'])}")
        print(f"Preferred topics size: {len(profile['preferred_topics'])}")
        
        return {
            "memory": memory,
            "emotion_analysis": emotion_analysis,
            "memory_evaluation": memory_evaluation,
            "profile_updates": {
                "known_vocabulary": profile["known_vocabulary"],
                "trigger_words": profile["trigger_words"],
                "preferred_topics": profile["preferred_topics"]
            }
        } 