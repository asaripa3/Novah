from utils.text_utils import normalize_list
import openai

class SanitizerAgent:
    def __init__(self, known_vocabulary, model, api_key, base_url):
        self.known_vocabulary = set(normalize_list(known_vocabulary))
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def sanitize(self, response: str) -> str:
        vocab_list = ", ".join(sorted(self.known_vocabulary))
        prompt = (
            f"You are a helpful assistant simplifying sentences for a child. "
            f"Follow these rules strictly:\n"
            f"1. Use only words from this vocabulary: [{vocab_list}]\n"
            f"2. Keep the exact same meaning as the original response\n"
            f"3. Do not change or assume preferences (e.g., if they want a car, don't change it to a truck)\n"
            f"4. Use a calm and friendly tone\n"
            f"5. Do not start with 'Hi' or repeat greetings\n"
            f"6. If a word is not in the vocabulary, explain it simply without changing the concept\n"
            f"7. Do not make assumptions about what the child likes or wants\n"
            f"8. Do not suggest anything unsafe or unrealistic\n"
            f"9. If you need to use a word not in the vocabulary, explain it in simple terms\n\n"
            f"Original response: \"{response}\"\n\n"
            f"Rewritten response:"
        )

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a careful assistant who simplifies language while preserving meaning. "
                    "Never change what the child wants or likes. "
                    "If they mention something specific (like a car), keep it as a car. "
                    "Only simplify the language, not the concepts."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent responses
        )
        return chat.choices[0].message.content.strip()