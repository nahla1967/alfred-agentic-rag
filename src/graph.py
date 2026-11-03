from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.retriever import guest_search, guests
from src.tools import get_weather, huggingface_search, web_search ,hub_stats


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatOllama(model="qwen2.5:7b", temperature=0)

tools = [guest_search, web_search, get_weather, huggingface_search ,hub_stats]
llm_with_tools = llm.bind_tools(tools)

guest_names = ", ".join(guest["name"] for guest in guests)

system_message = SystemMessage(
    content=(
        "You are Alfred, a helpful gala assistant. "
        "TOOL SELECTION RULES: "
        f"1. The gala guests are: {guest_names}. "
        "If the question mentions any of these people, "
        "always use guest_search, never web_search. "
        "2. Use get_weather for weather questions. "
        "3. Use huggingface_search for Hugging Face model questions. "
        
        "4. Use hub_stats when asked about the most downloaded "
"Hugging Face model by a specific author. "
"5e. Use web_search only for general questions "
        "that do not concern gala guests, weather, or Hugging Face models. "
        "Use only the information returned by the tools. "
        "Never invent information about guests."
    )
)


def agent_node(state: AgentState):
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_1"}}

    graph.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tunis?")]},
        config=config,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="What city did I ask about earlier?")]},
        config=config,
    )
    print("Alfred:", result["messages"][-1].content)