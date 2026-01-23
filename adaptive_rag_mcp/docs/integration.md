# Adaptive RAG MCP Integration Guide

This guide explains how to integrate the **Adaptive RAG MCP Server** into other projects, specifically how to configure it with MCP clients like **Claude Desktop** and how to use it programmatically.

## 1. Running the Server

Before any client can connect, the server must be running.

### Option A: Local Python Process
The most common way to run the server is as a local Python process.

1.  **Navigate to the project directory**:
    ```powershell
    cd d:\MCP\adaptive_rag_mcp
    ```

2.  **Activate Virtual Environment**:
    ```powershell
    # Windows
    .\.venv\Scripts\activate
    ```

3.  **Start the Server**:
    ```bash
    python -m src.server.main
    ```
    The server typically runs on `http://0.0.0.0:8000`.

### Option B: Docker (Recommended for Production)
If you have a `Dockerfile` setup (standard for MCP), you can build and run it as a container.

1.  **Build the Image**:
    ```bash
    docker build -t adaptive-rag .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8000:8000 -e ADAPTIVE_RAG_API_KEY=28ZIRfRtMgzIXb3kBN9kiYW5j4lof775viq2I2GshUQ adaptive-rag
    ```

3.  **Verify**:
    ```bash
    curl http://localhost:8000/health
    ```

## 2. Using with Cursor

Cursor has built-in support for MCP.

1.  **Open Settings**:
    Press `Ctrl+Shift+J` (Windows/Linux) or `Cmd+Shift+J` (Mac) to open Cursor Settings.

2.  **Navigate to MCP**:
    Go to `Features` -> `MCP` in the settings menu.

3.  **Add New Server**:
    Click the `+ Add new MCP server` button.

4.  **Configure**:
    *   **Name**: `adaptive-rag`
    *   **Type**: `command`
    *   **Command**: `D:/MCP/adaptive_rag_mcp/.venv/Scripts/python.exe`
    *   **Args**: `-m src.server.main`
    *   **Environment Variables**: 
        *   `ADAPTIVE_RAG_API_KEY=28ZIRfRtMgzIXb3kBN9kiYW5j4lof775viq2I2GshUQ`

## 3. Using with VS Code

Standard VS Code does not support MCP tools out of the box for general AI coding yet. However, you can use extensions.

### Via "Continue" Extension
If you use the [Continue](https://continue.dev/) extension:

1.  Open `config.json` in your `.continue` folder.
2.  Add to the `mcpServers` list:
    ```json
    {
      "name": "adaptive-rag",
      "command": "D:/MCP/adaptive_rag_mcp/.venv/Scripts/python.exe",
      "args": ["-m", "src.server.main"],
      "env": {
        "ADAPTIVE_RAG_API_KEY": "28ZIRfRtMgzIXb3kBN9kiYW5j4lof775viq2I2GshUQ"
      }
    }
    ```

## 4. Using with Antigravity (Gemini Code Assist)

If you are working within the Antigravity (Google IDX / Gemini Code Assist) environment:

1.  **Workspace Configuration**:
    Ensure the `adaptive_rag_mcp` folder is part of your active workspace.

2.  **Server Discovery**:
    Antigravity often automatically discovers MCP servers defined in standard locations or configuration files within the workspace (like `.vscode/mcp.json` or project-root configs, depending on version).

3.  **Manual Configuration**:
    If not discovered, you may need to register the server in your user settings or workspace settings file (similar to VS Code configuration) depending on the specific agent version you are running.

## 5. Configuring Claude Desktop

To use the Adaptive RAG capabilities within Claude Desktop:

1.  **Locate Configuration File**:
    *   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    *   **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

2.  **Add Server Configuration**:
    Add the `adaptive-rag` entry to the `mcpServers` object.

    ```json
    {
      "mcpServers": {
        "adaptive-rag": {
          "command": "D:/MCP/adaptive_rag_mcp/.venv/Scripts/python.exe",
          "args": [
            "-m",
            "src.server.main"
          ],
          "env": {
            "ADAPTIVE_RAG_API_KEY": "28ZIRfRtMgzIXb3kBN9kiYW5j4lof775viq2I2GshUQ"
          }
        }
      }
    }
    ```

3.  **Restart Claude Desktop**:
    Close and reopen Claude Desktop. You should see a plug icon indicating the tools are loaded.

## 6. Configuring a Generic MCP Client

If you are building your own MCP client or using another tool, the configuration generally requires:

*   **Transport**: Stdio (Standard Input/Output) is the default for local integrations.
*   **Command**: The python interpreter path.
*   **Arguments**: `["-m", "src.server.main"]`
*   **Environment Variables**: `ADAPTIVE_RAG_API_KEY` is required if authentication is enabled.

### Example (Python Client using `mcp` library)

```python
# Pseudo-code for a generic client
from mcp import Client, StdioServerParameters

server_params = StdioServerParameters(
    command="D:/MCP/adaptive_rag_mcp/.venv/Scripts/python.exe",
    args=["-m", "src.server.main"],
    env={"ADAPTIVE_RAG_API_KEY": "28ZIRfRtMgzIXb3kBN9kiYW5j4lof775viq2I2GshUQ"}
)

async with Client(server_params) as client:
    # List tools
    tools = await client.list_tools()
    
    # Call a tool
    result = await client.call_tool(
        "search", 
        arguments={"query": "machine learning"}
    )
```

## 7. HTTP API Usage (Non-MCP)

You can also use the server as a standard REST API for non-MCP projects.

*   **Base URL**: `http://localhost:8000`
*   **Authentication**: Header `X-API-Key: 28ZIRfRtMgzIXb3kBN9kiYW5j4lof775viq2I2GshUQ`

### Common Endpoints:

*   **POST** `/tools/search`
    ```json
    {
      "query": "your search query",
      "k": 5
    }
    ```

*   **POST** `/tools/ingest_document`
    ```json
    {
      "text": "Your text content here",
      "doc_id": "unique-id"
    }
    ```
