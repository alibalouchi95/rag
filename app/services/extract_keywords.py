from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# 1. Define the Pydantic Schema for structured output
class Keywords(BaseModel):
    """A list of extracted keywords."""

    keywords: List[str] = Field(
        ...,
        description="A list containing 5 to 10 most relevant, important keywords or key phrases.",
    )


def extract_keywords(text: str, model_name: str) -> List[str]:

    # Initialize the ChatOllama with the small model from main.py
    ollama_llm = ChatOllama(model=model_name, temperature=0)

    # Use LangChain to enforce JSON output using with_structured_output
    structured_llm = ollama_llm.with_structured_output(Keywords)

    # 2. Define the prompt template
    system_prompt = (
        "You are an expert keyword extraction tool. "
        "Your task is to analyze the provided text and extract the 5 to 10 most relevant, "
        "important keywords or key phrases. "
        "Return ONLY the keywords as a JSON list. Do NOT include explanations, validations, or reasoning. "
        "Your output MUST strictly follow the provided JSON schema."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Extract keywords from the following paragraph:\n\n{text}"),
        ]
    )

    # 3. Create the chain and invoke the model
    chain = prompt | structured_llm

    try:
        result: Keywords = chain.invoke({"text": text})

        # Ensure we got a valid Keywords object
        if not isinstance(result, Keywords):
            print(f"Warning: Unexpected result type: {type(result)}")
            return ""

        # Join keywords and enforce maximum length
        keyword_string = ", ".join(result.keywords)

        # CRITICAL: Enforce maximum length to prevent Milvus errors
        MAX_KEYWORD_LENGTH = 60000  # Safety buffer below 65535
        if len(keyword_string) > MAX_KEYWORD_LENGTH:
            print(
                f"Warning: Keywords truncated from {len(keyword_string)} to {MAX_KEYWORD_LENGTH} characters"
            )
            keyword_string = keyword_string[:MAX_KEYWORD_LENGTH]

        return keyword_string

    except Exception as e:
        print(f"Error during Ollama keyword extraction with {model_name}: {e}")
        return ""  # Return empty string instead of empty list
