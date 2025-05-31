from openai import OpenAI
from typing import Dict, List
import json
import re
from utils.text_utils import normalize_list

class QueryParserAgent:
    def __init__(self, model="llama3-70b-8192", api_key=None, base_url=None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tool_instructions = {
            "query_analysis": {
                "description": "Analyze the structure and intent of the query",
                "usage": "Use to understand what the user is asking",
                "output_format": "Returns query type and main intent"
            },
            "keyword_extraction": {
                "description": "Extract key concepts and important words",
                "usage": "Use to identify main topics and references",
                "output_format": "Returns list of relevant keywords"
            },
            "emotion_detection": {
                "description": "Detect emotional context and tone",
                "usage": "Use to understand user's emotional state",
                "output_format": "Returns primary emotion and emotional context"
            }
        }

    def parse_query(self, query: str) -> Dict:
        """
        Parse the user's query to extract key information and emotional context.
        
        Args:
            query: The user's input query
            
        Returns:
            Dict containing parsed information including keywords and emotion
        """
        # Thought process structure
        thought_process = f"""
        Thought Process:
        1. Query Analysis:
           - Input Text: {query}
           - Query Type: To be determined
           - Main Intent: To be identified
        
        2. Keyword Extraction:
           - Key Concepts: To be identified
           - Important References: To be noted
           - Contextual Elements: To be extracted
        
        3. Emotion Detection:
           - Emotional Tone: To be analyzed
           - Contextual Cues: To be identified
           - Emotional State: To be determined
        
        4. Final Integration:
           - Combine all analyses
           - Structure the output
           - Ensure completeness
        """

        # Tool usage instructions
        tool_instructions = """
        Available Tools:
        1. Query Analysis Tool:
           - Purpose: Understand query structure and intent
           - Usage: Analyze what the user is asking
           - Output: Query type and main intent

        2. Keyword Extraction Tool:
           - Purpose: Identify important concepts
           - Usage: Extract key topics and references
           - Output: List of relevant keywords

        3. Emotion Detection Tool:
           - Purpose: Understand emotional context
           - Usage: Analyze tone and emotional state
           - Output: Primary emotion and context
        """

        system_prompt = (
            f"{thought_process}\n\n"
            f"{tool_instructions}\n\n"
            "You are an assistant for parsing questions from a neurodiverse child. "
            "Your job is to extract: 1) key concepts or keywords, and 2) the emotion or tone. "
            "Include people, places, objects, and activities mentioned in the question — not just what is being asked. "
            "You will return a JSON with 'query_keywords' (as a list of words), "
            "'emotion' (e.g., curious, anxious, happy, confused), and 'raw_text'. "
            "Output only the JSON object and nothing else. "
            "Examples:\n"
            "\"I'm going to grandma's house for dinner. What is a chicken nugget?\" → ['grandma', 'house', 'dinner', 'chicken', 'nugget']\n"
            "\"Why is it raining today? Are we going to the doctor?\" → ['rain', 'today', 'doctor']"
        )

        user_prompt = f"User question: \"{query}\""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content
        try:
            # Clean up the response to ensure it's valid JSON
            content = content.replace('\n', ' ').replace('\r', '')
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {content}")
            return {
                "query_keywords": normalize_list(query.split()),
                "emotion": "neutral",
                "raw_text": query
            }