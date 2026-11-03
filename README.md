# Alfred – Agentic RAG Assistant

Alfred is an AI-powered gala assistant built with **Python, LangChain, and LangGraph**. It combines Retrieval-Augmented Generation (RAG), tool calling, and conversational memory to answer questions, retrieve guest information, access real-time data, and interact with external services.

## Features

- **Guest Information Retrieval (RAG):** Retrieves information about gala guests from a custom JSON dataset using BM25.
- **Web Search:** Searches the web for general information using DuckDuckGo.
- **Weather Information:** Retrieves current weather conditions for a given city using the Open-Meteo API.
- **Hugging Face Model Search:** Finds models based on machine learning tasks using the Hugging Face Hub.
- **Hub Statistics:** Identifies the most downloaded Hugging Face model by a specific author.
- **Conversational Memory:** Maintains conversation context across interactions using LangGraph's MemorySaver.

## Architecture

Alfred follows an agentic workflow built with LangGraph.

1. The user submits a question.
2. The LLM analyzes the request and decides whether a tool is needed.
3. LangGraph routes the request to the selected tool.
4. The tool executes and returns its result.
5. The LLM uses the result to generate a response.
6. MemorySaver maintains conversation history.

```text
              User Input
                  |
                  v
             Agent (LLM)
                  |
           Tool Selection
                  |
                  v
             ToolNode
                  |
     +------------+-------------+
     |            |             |
     v            v             v
 Guest RAG    Web Search    Weather API
     |            |             |
     +------------+-------------+
                  |
          Hugging Face Tools
                  |
                  v
             Agent (LLM)
                  |
                  v
             Final Answer
                  |
                  v
          Conversation Memory
```

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | Tool definition and LLM integration |
| LangGraph | Agent workflow orchestration |
| Ollama | Local LLM execution |
| Qwen 2.5 7B | Language model |
| BM25 | Keyword-based document retrieval |
| Hugging Face Hub | Model discovery and statistics |
| DuckDuckGo Search | Web search |
| Open-Meteo | Weather and geocoding APIs |

## Project Structure

```text
alfred-agentic-rag/
│
├── data/
│   └── guests.json
│
├── src/
│   ├── __init__.py
│   ├── retriever.py
│   ├── tools.py
│   └── graph.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone  https://github.com/nahla1967/alfred-agentic-rag.git
cd alfred-agentic-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Download Ollama from:

https://ollama.com/

Pull the required model:

```bash
ollama pull qwen2.5:7b
```

### 5. Run Alfred

```bash
python app.py
```

Type your questions in the terminal. To stop the assistant, type:

```text
exit
```

## Example Interactions

**Guest Information**
```text
You: Who is Hannibal Barca?

Alfred: Hannibal Barca was a Carthaginian general
who fought against Rome.
```

**Weather Information**
```text
You: What is the weather in Tunis?

Alfred: Weather in Tunis: Temperature: ...
```

**Hugging Face Model Search**
```text
You: Find models for text classification.

Alfred: Here are some Hugging Face models...
```

**Hub Statistics**
```text
You: What is the most downloaded Hugging Face model by facebook?

Alfred: The most downloaded model by Facebook is
facebook/contriever, with ... downloads.
```

**Conversational Memory**
```text
You: What city did I ask about earlier?

Alfred: You previously asked about Tunis.
```

## Learning Objectives

This project explores practical concepts in modern AI application development:

- Building RAG-powered applications.
- Creating custom tools for LLMs.
- Implementing tool calling and agentic workflows.
- Orchestrating agents using LangGraph.
- Integrating external APIs and Hugging Face services.
- Maintaining conversational context with checkpointing.

## Acknowledgements

This project was developed as a hands-on learning project inspired by the Hugging Face AI Agents course.

---

**Author:** Nahla Fersi  
**Project:** Alfred – Agentic RAG Assistant