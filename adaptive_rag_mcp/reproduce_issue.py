
import os
import sys
import asyncio
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("reproduce_issue")

# Add project root to path (addressing Issue B for this script context)
# In real usage, this should be handled by installing the package or setting PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Manually set environment variables to simulate the user's .env
# We are deliberately NOT filtering them here to test if the config fix works (Issue A)
# and then to see if we hit Issue C.
os.environ["ADAPTIVE_RAG_API_KEY"] = "dev-secret-key-change-in-production"
os.environ["GEMINI_API_KEY"] = "dummy-key" # Should be ignored now
os.environ["MCP_PROJECT_ROOT"] = current_dir

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    print("Starting reproduction script...")
    
    # Ensure subprocess can find 'src'
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{current_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = current_dir

    # Path to the python interpreter
    # Try to find the one in .venv first
    venv_python = os.path.join(current_dir, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        # Fallback to current sys.executable
        venv_python = sys.executable


    print(f"Using python: {venv_python}")
    
    server_params = StdioServerParameters(
        command=venv_python,
        args=["-m", "src.server.main"],
        env=os.environ.copy() # Pass all env vars including extras
    )

    print("Connecting to MCP server...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Connected successfully!")
                
                # List tools to verify functionality
                tools = await session.list_tools()
                print(f"Tools found: {[t.name for t in tools.tools]}")
                
    except Exception as e:
        print(f"Caught exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())
