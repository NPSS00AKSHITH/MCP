import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

ADAPTIVE_RAG_API_KEY = os.getenv("ADAPTIVE_RAG_API_KEY")
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND")
MCP_SERVER_ARGS = os.getenv("MCP_SERVER_ARGS", "").split()
MCP_PROJECT_ROOT = os.getenv("MCP_PROJECT_ROOT")

env = os.environ.copy()
env["ADAPTIVE_RAG_API_KEY"] = ADAPTIVE_RAG_API_KEY
env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ";" + str(MCP_PROJECT_ROOT)

# Remove restricted keys
for key in ["GEMINI_API_KEY", "MCP_SERVER_COMMAND", "MCP_SERVER_ARGS", "MCP_PROJECT_ROOT"]:
    if key in env:
        del env[key]

print(f"Running command: {MCP_SERVER_COMMAND} {MCP_SERVER_ARGS}")
print(f"Env PYTHONPATH: {env.get('PYTHONPATH')}")

try:
    # Run with pipes to capture output
    # Input needed for stdio server to not exit immediately if it waits for input
    # We provide empty input or minimal valid json
    input_str = '{"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0.1"}}, "id": 1}\n'
    
    process = subprocess.run(
        [MCP_SERVER_COMMAND] + MCP_SERVER_ARGS,
        env=env,
        input=input_str,
        capture_output=True,
        text=True,
        timeout=10
    )
    
    print(f"Exit Code: {process.returncode}")
    print("STDOUT:", process.stdout)
    print("STDERR:", process.stderr)

except subprocess.TimeoutExpired as e:
    print("Timeout expired!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
except Exception as e:
    print(f"Error: {e}")
