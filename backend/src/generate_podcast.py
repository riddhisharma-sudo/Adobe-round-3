from .chat_with_llm import get_llm_response
from .prompts import PODCAST_GENERATION_PROMPT
import os
import json
import re

def generate_podcast(selected_insights : str, output_path: str = None):
    """
    Generate podcast JSON using the LLM from current section and related section
    """
    prompt = PODCAST_GENERATION_PROMPT.format(selected_insights=selected_insights)

    messages = [
        {"role": "system", "content": "You are tasked with creating a natural, engaging 2-5 minute podcast conversation between two hosts based on the provided JSON data containing document insights and extracted sections."},
        {"role": "user", "content": prompt}
    ]

    try:
        reply = get_llm_response(messages)

        clean = re.sub(r'```(?:[^\n]*)\n?(.*?)```', r'\1', reply, flags=re.S).strip()

        data = json.loads(clean)

        return data
        
    except Exception as e:
        print("Error:", str(e))
        raise e


if __name__ == "__main__":

    current_dir = os.path.dirname(__file__)

    input_json_path = os.path.join(current_dir, "..", "Adobe_Round_1B", "output", "input_output.json")
    insights_path = os.path.join(current_dir, "output", "insights.json")

    if not os.path.exists(input_json_path):
        print(f"Input JSON file not found at {input_json_path}")
        exit(1)

    generate_podcast(input_json_path, insights_path)
