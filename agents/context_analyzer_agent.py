from openai import OpenAI
from typing import Dict, List, Tuple
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ContextAnalyzer')

class ContextAnalyzerAgent:
    def __init__(self, model="llama3-70b-8192", api_key=None, base_url=None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized ContextAnalyzerAgent with model: {model}")

    def analyze_context(self, query: str, chat_history: List[str]) -> Tuple[bool, List[str]]:
        """
        Analyze whether the current query needs chat history context.
        
        Args:
            query: The current user query
            chat_history: List of previous chat messages
            
        Returns:
            Tuple containing:
            - Boolean indicating if chat history should be used
            - List of relevant chat history messages to use (empty if none needed)
        """
        logger.info(f"\n{'='*50}\nAnalyzing context for new query")
        logger.info(f"Query: {query}")
        logger.info(f"Chat history length: {len(chat_history)} messages")

        if not chat_history:
            logger.info("No chat history available, skipping context analysis")
            return False, []

        # Create a prompt for the LLM to analyze the context
        system_prompt = (
            "You are an expert at analyzing conversation context. "
            "Your job is to determine if the current query is a follow-up question "
            "or needs context from previous messages. "
            "Return a JSON with:\n"
            "- needs_context: boolean indicating if chat history is needed\n"
            "- relevant_messages: list of relevant message indices (0-based) from chat history\n"
            "- reasoning: brief explanation of your decision\n"
            "Only return the JSON object, no other text."
        )

        # Format chat history for the prompt
        formatted_history = "\n".join([f"{i}: {msg}" for i, msg in enumerate(chat_history)])
        logger.info("Formatted chat history for analysis:")
        logger.info(formatted_history)
        
        user_prompt = (
            f"Chat History:\n{formatted_history}\n\n"
            f"Current Query: {query}\n\n"
            "Analyze if this query needs context from the chat history."
        )

        logger.info("Sending request to LLM for context analysis...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        try:
            content = response.choices[0].message.content.strip()
            logger.info("Received response from LLM")
            
            # Clean up the response to ensure it's valid JSON
            content = content.replace('\n', ' ').replace('\r', '')
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            analysis = json.loads(content)
            logger.info("Context analysis result:")
            logger.info(f"Needs context: {analysis.get('needs_context', False)}")
            logger.info(f"Reasoning: {analysis.get('reasoning', 'No reasoning provided')}")
            
            # Get relevant messages if needed
            relevant_messages = []
            if analysis.get("needs_context", False):
                indices = analysis.get("relevant_messages", [])
                relevant_messages = [chat_history[i] for i in indices if i < len(chat_history)]
                logger.info(f"Selected {len(relevant_messages)} relevant messages:")
                for msg in relevant_messages:
                    logger.info(f"- {msg}")
            else:
                logger.info("No relevant messages selected")
            
            logger.info(f"{'='*50}\n")
            return analysis.get("needs_context", False), relevant_messages
            
        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            logger.error(f"Raw LLM response: {content}")
            logger.info(f"{'='*50}\n")
            return False, [] 