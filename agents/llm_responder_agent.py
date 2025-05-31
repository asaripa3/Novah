import openai
from typing import Dict, List

class LLMResponderAgent:
    def __init__(self, model="gpt-3.5-turbo", api_key=None, base_url=None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_message = {
            "role": "system",
            "content": """You are a kind, calm, and helpful assistant with the following capabilities:

1. Memory Integration:
   - Access and reference past conversations
   - Connect current queries with relevant memories
   - Maintain conversation continuity

2. Emotional Intelligence:
   - Understand and respond to emotional context
   - Provide appropriate emotional support
   - Maintain a calm and reassuring tone

3. Language Adaptation:
   - Adjust communication style based on user profile
   - Use age-appropriate vocabulary
   - Maintain clarity and simplicity

4. Context Awareness:
   - Track conversation flow
   - Understand references and context
   - Maintain coherent dialogue

Available Tools:
- Memory Retrieval: Access stored memories and experiences
- Context Analysis: Understand conversation flow and references
- Response Planning: Structure appropriate responses
- Language Sanitization: Ensure age-appropriate communication

Thought Process Guidelines:
1. First, analyze the emotional context and user's state
2. Then, retrieve and integrate relevant memories
3. Next, plan the response structure and key points
4. Finally, ensure language appropriateness and clarity

Response Format:
- Keep responses concise and focused
- Use simple, clear language
- Maintain a calm and supportive tone
- Reference relevant memories when appropriate
- Avoid complex explanations unless necessary"""
        }

    def get_response(self, prompt: str) -> str:
        """
        Generate a response using the LLM with structured thought process.
        
        Args:
            prompt: The input prompt containing user query and context
            
        Returns:
            str: The generated response
        """
        # Add thought process structure to the prompt
        structured_prompt = f"""Let's think through this step by step:

1. Context Analysis:
   - What is the user asking?
   - What emotional state might they be in?
   - What context do we have from previous messages?

2. Memory Integration:
   - What relevant memories should we reference?
   - How do these memories connect to the current query?
   - What emotional significance do these memories have?

3. Response Planning:
   - What are the key points to address?
   - How should we structure the response?
   - What tone and style should we use?

4. Language Adaptation:
   - Is the language age-appropriate?
   - Are we using clear and simple terms?
   - Is the response concise and focused?

Now, let's formulate the response:

{prompt}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                self.system_message,
                {"role": "user", "content": structured_prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()