from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
@tool
def guest_search(query: str) -> str:
    """Search for information about gala guests."""
    return "Marie Curie was a scientist who pioneered the study of radioactivity."

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)

llm_with_tools = llm.bind_tools([guest_search])

response = llm_with_tools.invoke(
    "Who is Marie Curie? Use the guest_search tool."
)
tool_call = response.tool_calls[0]

tool_result = guest_search.invoke(tool_call["args"])

print("Tool result:", tool_result)

messages = [
    ("user", "Who is Marie Curie?"),
    response,
    ToolMessage(
        content=tool_result,
        tool_call_id=tool_call["id"]
    )
]

final_answer = llm.invoke(messages)

print("Alfred:", final_answer.content)

print(response)