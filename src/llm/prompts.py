"""Model config and prompt templates for LLM calls."""

# MODEL = "llama-3.3-70b-versatile"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


CONTEXTUALIZER_SYSTEM_PROMPT = """You write brief, factual contextual prefixes for text chunks from academic papers. Given the paper's abstract and one chunk from that paper, write a 1–2 sentence prefix (max 100 words) describing what the chunk is about in relation to the paper's main work. Output ONLY the prefix — no preamble, no labels, no quotes."""
