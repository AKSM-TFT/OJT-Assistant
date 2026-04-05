import os
from langchain_openai import ChatOpenAI
 
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4-5"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    max_tokens=4096,
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Resume AI"),
    },
)