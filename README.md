# AI Business Analyst Agent

This project implements an AI agent using LangGraph to perform daily business performance analysis. It calculates key metrics from sales and cost data and uses an LLM to generate an actionable report.

## Features

- **Modular Design:** Built with LangGraph for a clear and scalable workflow.
- **Flexible LLM Integration:** Easily configurable to use any OpenAI-compatible API (like OpenRouter or a local model) via environment variables.
- **Reliable, Structured Output:** Uses Pydantic to ensure the LLM's response is always a valid JSON object.
- **Built-in Validation:** Comes with a comprehensive `unittest` suite to ensure logical correctness.

## How to Run

### 1. Prerequisites

- Python 3.8+
- An API key from an LLM provider (e.g., [OpenRouter.ai](https://openrouter.ai/))
- A LangSmith API key for tracing.

### 2. Setup

1.  **Clone the repository (if on GitHub) or unzip the project folder.**

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure your environment:**

    - Make a `.env` file.
    - Open the `.env` file and fill in your details:

    ```ini
    # Your API key from the provider
    # I used openrouter api key
    LLM_API_KEY="YOUR_API_KEY_HERE"

    # The base URL of the API. For OpenRouter: [https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)
    LLM_BASE_URL="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)"

    # The model identifier, e.g., "deepseek/deepseek-chat"
    LLM_MODEL_NAME="deepseek/deepseek-chat:free" # if you are using openrouter

    # --- LangSmith studio Tracing ---
    LANGCHAIN_TRACING_V2="true" # Set to "true" to enable
    LANGCHAIN_API_KEY="YOUR_LANGSMITH_KEY_HERE"
    LANGCHAIN_PROJECT="Business Agent Evaluation"
    ```

### 3. Execution

Run the agent directly from your terminal:

```bash
python main.py
```
