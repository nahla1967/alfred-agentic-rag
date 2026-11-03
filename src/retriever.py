import json

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool

with open("data/guests.json", "r", encoding="utf-8") as file:
    guests = json.load(file)

documents = [
    Document(
        page_content=f"{guest['name']}. {guest['description']}",
        metadata={
            "name": guest["name"],
            "relation": guest["relation"],
            "email": guest["email"],
        },
    )
    for guest in guests
]

retriever = BM25Retriever.from_documents(documents, k=1)


@tool
def guest_search(query: str) -> str:
    """Search the local gala guest database.
    Use this tool whenever the question mentions a gala guest,
    including their name, background, relation, or email.
    """
    results = retriever.invoke(query)
    return "\n".join(document.page_content for document in results)


if __name__ == "__main__":
    print(f"Loaded {len(documents)} documents")
    for question in [
        "Who is Marie Curie?",
        "Tell me about Hannibal Barca",
        "What is Albert Einstein known for?",
        "Who is Isaac Newton?",
    ]:
        print(f"\nQuestion: {question}")
        print(guest_search.invoke({"query": question}))