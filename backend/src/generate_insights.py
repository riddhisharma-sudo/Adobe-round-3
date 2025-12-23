from .chat_with_llm import get_llm_response
from .prompts import INSIGHTS_BULB_PROMPT
import os
import json
import re

def generate_insights(related_json_path : str):
    """
    Generate insights using the LLM.
    """

    with open(related_json_path, "r") as f:
        config = json.load(f)

    if "selected_content" in config and "similar_chunks" in config:
        selected_content = config["selected_content"]
        similar_chunks = config.get("similar_chunks", [])
        
        # Create a structure similar to the old format for the prompt
        formatted_data = {
            "current_section": {
                "title": "Selected Content",
                "content": selected_content,
                "source": "User Selection"
            },
            "related_sections": []
        }
        
        # Add similar chunks as related sections
        for i, chunk in enumerate(similar_chunks):
            formatted_data["related_sections"].append({
                "title": chunk.get("heading", f"Section {i+1}"),
                "content": chunk["text"],
                "source": f"{chunk['doc']} (Page {chunk['page']})",
                "similarity_score": chunk.get("similarity_score", 0.0)
            })
        
        if "metadata" in config:
            formatted_data["metadata"] = config["metadata"]
        
        config_str = json.dumps(formatted_data, indent=2, ensure_ascii=False)
    else:
        # Old format: Use as is
        config_str = json.dumps(config, indent=2, ensure_ascii=False)
    
    prompt = INSIGHTS_BULB_PROMPT.format(selected_insights=config_str)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates valuable insights from document content."},
        {"role": "user", "content": prompt}
    ]

    try:
        reply = get_llm_response(messages)

        clean = re.sub(r'```(?:[^\n]*)\n?(.*?)```', r'\1', reply, flags=re.S).strip()

        # Convert string to Python object
        data = json.loads(clean)

        return data
    except Exception as e:
        print("Error:", str(e))
        return None


if __name__ == "__main__":

    current_dir = os.path.dirname(__file__)

    input_json_path = os.path.join(current_dir, "..", "Adobe_Round_1B", "output", "input_output.json")


    generate_insights(input_json_path)
