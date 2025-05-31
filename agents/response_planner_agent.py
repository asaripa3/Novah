class ResponsePlannerAgent:
    def __init__(self, max_words=20, context_analyzer=None):
        self.max_words = max_words
        self.context_analyzer = context_analyzer
        self.tool_instructions = {
            "context_analysis": {
                "description": "Analyze conversation context and determine if chat history is needed",
                "usage": "Use when determining if previous messages are relevant to current query",
                "output_format": "Returns boolean and list of relevant messages"
            },
            "memory_integration": {
                "description": "Integrate relevant memories into the response",
                "usage": "Use when user's query relates to past experiences",
                "output_format": "Returns formatted memory text with emotional context"
            },
            "response_structure": {
                "description": "Structure the response for clarity and coherence",
                "usage": "Use to organize response points and maintain flow",
                "output_format": "Returns structured response with clear sections"
            }
        }

    def build_prompt(self, query: str, memory_text: str, profile: dict, dialogue_context: str = "") -> str:
        from utils.text_utils import normalize_list

        if "name" not in profile or "mental_age" not in profile:
            raise ValueError("Profile must include 'name' and 'mental_age'")
        name = profile["name"]
        age = profile["mental_age"]

        # Thought process structure
        thought_process = f"""
        Thought Process:
        1. Context Analysis:
           - User Profile: {name} is {age} years old
           - Query Analysis: {query}
           - Emotional Context: To be determined by psychiatrist agent
        
        2. Memory Integration:
           - Available Memory: {memory_text}
           - Memory Relevance: To be evaluated
           - Emotional Connection: To be assessed
        
        3. Response Planning:
           - Key Points: To be identified
           - Structure: To be determined
           - Tone: Calm and supportive
        
        4. Language Adaptation:
           - Age Appropriateness: {age} years old
           - Vocabulary Level: To be adjusted
           - Clarity: To be ensured
        """

        query_keywords = set(normalize_list(query.split()))
        memory_keywords = set(normalize_list(memory_text.split()))
        shared_keywords = query_keywords.intersection(memory_keywords)

        memory_clause = ""
        if memory_text and not memory_text.startswith("No specific") and shared_keywords:
            memory_clause = f'If helpful, use this memory: "{memory_text}"\n'

        # Analyze if we need chat history context
        context_clause = ""
        if dialogue_context and self.context_analyzer:
            chat_history = dialogue_context.split("\n")
            needs_context, relevant_messages = self.context_analyzer.analyze_context(query, chat_history)
            if needs_context and relevant_messages:
                context_clause = f"Recent conversation: {' '.join(relevant_messages)}\n"

        # Tool usage instructions
        tool_instructions = """
        Available Tools:
        1. Context Analysis Tool:
           - Purpose: Determine if chat history is needed
           - Usage: When query might reference previous messages
           - Output: Boolean and relevant message list

        2. Memory Integration Tool:
           - Purpose: Connect current query with past experiences
           - Usage: When query relates to previous interactions
           - Output: Formatted memory text with context

        3. Response Structure Tool:
           - Purpose: Organize response for clarity
           - Usage: Always, to ensure coherent responses
           - Output: Structured response format
        """

        return (
            f"{thought_process}\n\n"
            f"{tool_instructions}\n\n"
            f"Response Guidelines:\n"
            f"- {name} is a {age}-year-old boy with autism\n"
            f"- Use a calm tone and no more than {self.max_words} words\n"
            f"- Do not start with 'Hi' or repeat greetings unless the user starts with one\n"
            f"- {memory_clause}"
            f"- {context_clause}"
            f"Question: {query}\n"
        )