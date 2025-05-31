from openai import OpenAI
from typing import Dict, List
from utils.text_utils import normalize_list

class SanitizerAgent:
    def __init__(self, known_vocabulary: List[str], model="llama3-70b-8192", api_key=None, base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.known_vocabulary = set(normalize_list(known_vocabulary))
        self.tool_instructions = {
            "vocabulary_check": {
                "description": "Ensure age-appropriate language",
                "usage": "Check each word against known vocabulary",
                "output_format": "Returns list of words needing simplification"
            },
            "language_simplification": {
                "description": "Simplify complex language while maintaining meaning",
                "usage": "When vocabulary check identifies complex words",
                "output_format": "Returns simplified text with explanations"
            },
            "tone_adjustment": {
                "description": "Ensure appropriate tone and style",
                "usage": "To maintain calm and supportive communication",
                "output_format": "Returns text with adjusted tone"
            }
        }

    def sanitize(self, response: str) -> str:
        """Alias for sanitize_response to maintain backward compatibility."""
        return self.sanitize_response(response)

    def sanitize_response(self, response: str) -> str:
        """Sanitize the response to be appropriate for a child."""
        print("\n[Sanitizer Agent] Processing response")
        print(f"Original response: {response}")
        print(f"Known vocabulary size: {len(self.known_vocabulary)}")

        # Thought process structure (internal only)
        thought_process = f"""
        Thought Process:
        1. Vocabulary Analysis:
           - Known Vocabulary: {len(self.known_vocabulary)} words
           - Complex Words: To be identified
           - Age-Appropriate Terms: To be selected
        
        2. Language Simplification:
           - Complex Concepts: To be simplified
           - Meaning Preservation: To be ensured
           - Clarity: To be maintained
        
        3. Tone Assessment:
           - Current Tone: To be evaluated
           - Emotional Impact: To be considered
           - Supportiveness: To be ensured
        
        4. Final Review:
           - Vocabulary Check: To be performed
           - Clarity Check: To be performed
           - Tone Check: To be performed
        """

        # Tool usage instructions (internal only)
        tool_instructions = """
        Available Tools:
        1. Vocabulary Check Tool:
           - Purpose: Ensure age-appropriate language
           - Usage: Check each word against known vocabulary
           - Output: List of words needing simplification

        2. Language Simplification Tool:
           - Purpose: Simplify complex language
           - Usage: When vocabulary check identifies complex words
           - Output: Simplified text with explanations

        3. Tone Adjustment Tool:
           - Purpose: Ensure appropriate tone
           - Usage: To maintain calm and supportive communication
           - Output: Text with adjusted tone
        """

        prompt = (
            f"{thought_process}\n\n"
            f"{tool_instructions}\n\n"
            "You are an expert in child psychology and language development. "
            "Rewrite this response to be appropriate for a 10-year-old child.\n\n"
            "Guidelines:\n"
            "1. Use only words from the known vocabulary list\n"
            "2. Keep sentences short and clear\n"
            "3. Maintain a calm and supportive tone\n"
            "4. Avoid complex concepts or explanations\n"
            "5. Use simple, direct language\n"
            "6. Keep the original meaning\n"
            "7. Make it engaging and friendly\n\n"
            f"Known vocabulary: {', '.join(sorted(self.known_vocabulary))}\n\n"
            f"Original response: {response}\n\n"
            "Return ONLY the rewritten response, no other text or explanations."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in child psychology and language development. Return only the rewritten response."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        sanitized_response = response.choices[0].message.content.strip()
        print(f"Sanitized response: {sanitized_response}")
        return sanitized_response