from openai import OpenAI
from typing import Dict
import json
import re
from utils.text_utils import normalize_list

class QueryParserAgent:
    def __init__(self, model="llama3-70b-8192", api_key=None, base_url=None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def parse_query(self, query: str) -> Dict:
        system_prompt = (
            "You are an assistant for parsing questions from a neurodiverse child. "
            "Your job is to extract: 1) key concepts or keywords, and 2) the emotion or tone. "
            "Include people, places, objects, and activities mentioned in the question — not just what is being asked. "
            "You will return a JSON with 'query_keywords' (as a list of words), "
            "'emotion' (e.g., curious, anxious, happy, confused), and 'raw_text'. "
            "Output only the JSON object and nothing else. "
            "Examples:\n"
            "\"I’m going to grandma’s house for dinner. What is a chicken nugget?\" → ['grandma', 'house', 'dinner', 'chicken', 'nugget']\n"
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
            content = content.strip().replace("\r", "").replace("\n", " ")
            if content.endswith('""}'):
                content = content.replace('""}', '"}')
            if content.endswith('"}"}'):
                content = content[:-3] + '"}'

            json_match = re.search(r"\{.*\}", content)
            if json_match:
                parsed = json.loads(json_match.group())
                import itertools
                tokens = list(itertools.chain.from_iterable(
                    word.split() for word in parsed.get("query_keywords", [])
                ))
                parsed["query_keywords"] = normalize_list(tokens)
                return parsed
            else:
                print(f"[LLM RAW RESPONSE - NO JSON FOUND]\n{content}")
                raise ValueError("No JSON object found in model response.")
        except Exception as e:
            print(f"[LLM RAW RESPONSE - PARSE ERROR]\n{content}")
            print(f"Error parsing response: {e}")
            return {
                "query_keywords": [],
                "emotion": "unknown",
                "raw_text": query
            }