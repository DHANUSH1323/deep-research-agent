import json
from framework.tools import get_all_schemas, tool, get_tool, _TOOL_REGISTRY

@tool
def search_corpus(query: str, top_k: int = 5) -> str:
    """Search the research corpus for relevant papers."""
    # Placeholder implementation
    return f"Searching for '{query}' and returning top {top_k} results."

@tool
def get_paper_full_text(arxiv_id: str) -> str:
    """Retrieve the full text of a paper given its arXiv ID."""
    # Placeholder implementation
    return f"Full text for paper with arXiv ID: {arxiv_id}"

print("Registry keys:", list(_TOOL_REGISTRY.keys()))

print("\nschema attached to search_corpus:")
print(search_corpus.schema)

print("\nLooked-up call result:")
looked_up = get_tool("search_corpus")
print(looked_up(query="positional encoding"))

print("\nAll schemas:")
print(json.dumps(get_all_schemas(), indent=2))