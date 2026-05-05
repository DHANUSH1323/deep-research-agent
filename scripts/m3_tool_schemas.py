import json
from src.agents._tools import tool_to_schema

def search_corpus(query: str, top_k: int = 5):

    """Search the research corpus for relevant papers."""
    # Placeholder implementation
    pass

print(json.dumps(tool_to_schema(search_corpus), indent=2))

