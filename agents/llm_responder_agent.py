import openai

class LLMResponderAgent:
    def __init__(self, model="gpt-3.5-turbo", api_key=None, base_url=None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def get_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a kind, calm, and helpful assistant who answers questions with a gentle tone and incorporates the user's past experiences if relevant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()