from langchain_core.messages import HumanMessage

from src.graph import graph

config = {"configurable": {"thread_id": "alfred_1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # Keep only messages from the current turn (after the last human message)
    current_turn = []
    for message in reversed(response["messages"]):
        if message.type == "human":
            break
        current_turn.append(message)

    for message in reversed(current_turn):
        if message.type == "tool":
            print("TOOL NAME:", message.name)
            print("TOOL RESULT:", message.content)

    print("Alfred:", response["messages"][-1].content)