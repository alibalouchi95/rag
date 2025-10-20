import json
from huggingface_hub import InferenceClient
from typing import List, Union


def extract_keywords(text, hf_api_key) -> List[str]:
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=hf_api_key,
    )

    # Cleaned-up System Prompt: Keeping the instruction for the 'keywords' key
    system_prompt = (
        "You are an expert keyword extraction tool. "
        "Your task is to analyze the provided text and extract the 5 to 10 most relevant, "
        "important keywords or key phrases. "
        "Your output MUST be a valid JSON object with a single key **'keywords'** "
        "whose value is an array of 5 to 10 strings. Do not include any text outside of the JSON."
    )

    user_prompt = f"Extract keywords from the following paragraph:\n\n{text}"

    # FIX: Remove the strict 'response_format' parameter to avoid the 400 Bad Request / Unexpected EOS error
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # response_format={"type": "json_object"}, # This line was removed
    )

    response_content = ""
    try:
        response_content = completion.choices[0].message.content
        # 3. Parse the JSON string into a Python dict/list
        parsed_output: Union[dict, list] = json.loads(response_content)

        final_keywords: List[str] = []

        # --- ROBUST FIX REMAINS ---
        # Define the possible keys the LLM might use for the list (since it can be inconsistent)
        POSSIBLE_KEYS = ["keywords", "content", "phrases", "data"]

        if isinstance(parsed_output, dict):
            # Iterate through possible keys to find the list
            for key in POSSIBLE_KEYS:
                if key in parsed_output and isinstance(parsed_output[key], list):
                    final_keywords = parsed_output[key]
                    break  # Found the list, stop searching
        elif isinstance(parsed_output, list):
            # If the model returns the list directly
            final_keywords = parsed_output
        else:
            print(
                f"Warning: LLM returned unexpected JSON structure: {response_content}"
            )
            return []

        # Now, validate the extracted list of keywords
        if isinstance(final_keywords, list) and all(
            isinstance(k, str) for k in final_keywords
        ):
            return final_keywords
        else:
            print(
                f"Warning: Extracted value is not a valid list of strings: {final_keywords}"
            )
            return []
        # --- ROBUST FIX ENDS ---

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}\nRaw Response: {response_content}")
        return []
    except Exception as e:
        print("Error while extracting keywords:", e)
        return []
