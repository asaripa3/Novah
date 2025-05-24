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
            f"Rewrite the following response using only this vocabulary: [{vocab_list}]. "
            f"Use a calm and friendly tone. Do not start with 'Hi' or repeat greetings. "
            f"Keep the meaning. If a word or concept is not in the vocabulary, rephrase it simply "
            f"(e.g., 'Disneyland' → 'fun place'). Do not make up unrelated things. "
            f"Avoid suggesting anything unsafe or unrealistic.\n\n"
            f"Original response: \"{response}\"\n\n"
            f"Rewritten response:"
        )

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You simplify sentences for clarity and child-friendly understanding."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return chat.choices[0].message.content.strip()