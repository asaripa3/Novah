class ResponsePlannerAgent:
    def __init__(self, max_words=20, context_analyzer=None):
        self.max_words = max_words
        self.context_analyzer = context_analyzer

    def build_prompt(self, query: str, memory_text: str, profile: dict, dialogue_context: str = "") -> str:
        from utils.text_utils import normalize_list

        if "name" not in profile or "mental_age" not in profile:
            raise ValueError("Profile must include 'name' and 'mental_age'")
        name = profile["name"]
        age = profile["mental_age"]

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

        return (
            f"{name} is a {age}-year-old boy with autism.\n"
            f"Use a calm tone and no more than {self.max_words} words.\n"
            f"Do not start with 'Hi' or repeat greetings unless the user starts with one.\n"
            f"{memory_clause}"
            f"{context_clause}"
            f"Question: {query}\n"
        )