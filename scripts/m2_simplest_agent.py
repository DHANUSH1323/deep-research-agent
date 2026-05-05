from src.llm.anthropic_client import get_anthropic_client


from src.config import SUBAGENT_MODEL


question = "What is retrieval-augmented generation in 3 sentences?"
client = get_anthropic_client()

response = client.messages.create(model=SUBAGENT_MODEL, messages=[{"role": "user", "content": question}], max_tokens = 16384)

print("response.stop_reason:", response.stop_reason)
print("response.content:", response.content)
print("response.content[0].text:", response.content[0].text)
print("response.usage:", response.usage)