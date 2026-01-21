"""Example script to test the MCP server.

Run the server first:
    python -m src.server.main

Then run this script:
    python examples/test_server.py
"""

import httpx
import json

BASE_URL = "http://localhost:8000"
API_KEY = "dev-secret-key-change-in-production"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def test_health():
    """Test health endpoint (no auth required)."""
    print("=" * 50)
    print("Testing: GET /health")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_list_tools():
    """Test listing all tools."""
    print("\n" + "=" * 50)
    print("Testing: GET /tools")
    response = httpx.get(f"{BASE_URL}/tools", headers=HEADERS)
    print(f"Status: {response.status_code}")
    tools = response.json()
    print(f"Available tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:50]}...")
    return response.status_code == 200


def test_search_tool():
    """Test calling the search tool."""
    print("\n" + "=" * 50)
    print("Testing: POST /tools/search")
    
    request_data = {
        "query": "What is retrieval augmented generation?",
        "k": 3,
        "collection_id": "default"
    }
    
    response = httpx.post(
        f"{BASE_URL}/tools/search",
        headers=HEADERS,
        json=request_data,
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result['success']}")
    print(f"Results count: {len(result['result']['results'])}")
    print("Sample result:")
    print(json.dumps(result['result']['results'][0], indent=2))
    return response.status_code == 200


def test_decide_retrieval_tool():
    """Test the decide_retrieval tool."""
    print("\n" + "=" * 50)
    print("Testing: POST /tools/decide_retrieval")
    
    request_data = {
        "query": "What are the key differences between dense and sparse retrieval methods in modern RAG systems, and when should each be used?"
    }
    
    response = httpx.post(
        f"{BASE_URL}/tools/decide_retrieval",
        headers=HEADERS,
        json=request_data,
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Strategy: {result['result']['strategy']}")
    print(f"Complexity: {result['result']['complexity']}")
    print(f"Reasoning: {result['result']['reasoning']}")
    return response.status_code == 200


def test_validation_error():
    """Test that validation errors are properly returned."""
    print("\n" + "=" * 50)
    print("Testing: Validation Error (missing required field)")
    
    # Missing required 'query' field
    request_data = {"k": 5}
    
    response = httpx.post(
        f"{BASE_URL}/tools/search",
        headers=HEADERS,
        json=request_data,
    )
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json()['detail']}")
    return response.status_code == 422


def test_auth_error():
    """Test that auth errors are properly returned."""
    print("\n" + "=" * 50)
    print("Testing: Auth Error (invalid API key)")
    
    response = httpx.get(
        f"{BASE_URL}/tools",
        headers={"X-API-Key": "wrong-key"},
    )
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json()['detail']}")
    return response.status_code == 403


def main():
    """Run all tests."""
    print("\n🚀 Adaptive RAG MCP Server - Test Suite\n")
    
    results = []
    try:
        results.append(("Health Check", test_health()))
        results.append(("List Tools", test_list_tools()))
        results.append(("Search Tool", test_search_tool()))
        results.append(("Decide Retrieval", test_decide_retrieval_tool()))
        results.append(("Validation Error", test_validation_error()))
        results.append(("Auth Error", test_auth_error()))
    except httpx.ConnectError:
        print("\n❌ Connection Error: Is the server running?")
        print("   Start it with: python -m src.server.main")
        return
    
    print("\n" + "=" * 50)
    print("Test Results:")
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))


if __name__ == "__main__":
    main()
