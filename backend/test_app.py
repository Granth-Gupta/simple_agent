import pytest
import pytest_asyncio
import httpx
import asyncio
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0  # Increased timeout for initialization


class TestBackendAPI:
    """Test suite for the AI Agent Backend API"""
    
    @pytest_asyncio.fixture(scope="class", autouse=True)
    async def wait_for_server_startup(self):
        """Wait for server to fully initialize before running tests"""
        print("Waiting for server to initialize...")
        await asyncio.sleep(3)  # Give server time to start up
        
        # Test server availability
        max_retries = 10
        server_ready = False
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
                    response = await client.get("/health")
                    if response.status_code == 200:
                        print(f"Server ready after {attempt + 1} attempts")
                        server_ready = True
                        break
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt < max_retries - 1:
                    print(f"Server not ready, attempt {attempt + 1}/{max_retries}")
                    await asyncio.sleep(2)
                else:
                    print("Server not available for testing")
        
        if not server_ready:
            pytest.skip("Server not available - make sure to run 'python app.py' first")

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test the health check endpoint"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = await client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert "status" in data
            assert "tools_available" in data
            assert isinstance(data["status"], str)
            assert isinstance(data["tools_available"], int)
            
            # Status should be either "healthy" or "initializing"
            assert data["status"] in ["healthy", "initializing"]
            
            # Tools available should be non-negative
            assert data["tools_available"] >= 0
            
            print(f"Health check passed: {data}")

    @pytest.mark.asyncio
    async def test_tools_endpoint(self):
        """Test the tools endpoint returns available tools"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = await client.get("/tools")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert "tools" in data
            assert isinstance(data["tools"], list)
            
            # Should have at least some tools (firecrawl tools)
            assert len(data["tools"]) > 0, "No tools returned from API"
            
            # All tools should be strings
            for tool in data["tools"]:
                assert isinstance(tool, str)
                assert len(tool) > 0
            
            # Should contain expected firecrawl tools
            expected_tools = ["firecrawl_scrape", "firecrawl_search", "firecrawl_extract"]
            found_tools = [tool for tool in expected_tools if tool in data["tools"]]
            assert len(found_tools) > 0, f"Expected tools not found. Available: {data['tools']}"
            
            print(f"Tools endpoint passed: {len(data['tools'])} tools available")

    @pytest.mark.asyncio
    async def test_chat_empty_message(self):
        """Test chat endpoint with empty message returns 400"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = await client.post("/chat", json={"message": "", "history": []})
            
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Empty message"
            
            print("Empty message test passed")

    @pytest.mark.asyncio
    async def test_chat_whitespace_message(self):
        """Test chat endpoint with whitespace-only message returns 400"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = await client.post("/chat", json={"message": "   \n\t  ", "history": []})
            
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Empty message"
            
            print("Whitespace message test passed")

    @pytest.mark.asyncio
    async def test_chat_invalid_json(self):
        """Test chat endpoint with invalid JSON structure"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
            # Missing required field
            response = await client.post("/chat", json={"history": []})
            
            assert response.status_code == 422  # Validation error
            
            print("Invalid JSON test passed")

    @pytest.mark.asyncio
    async def test_chat_valid_simple_message(self):
        """Test chat endpoint with a simple valid message"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:  # Longer timeout for AI response
            response = await client.post("/chat", json={
                "message": "Hello, what can you help me with?",
                "history": []
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert "success" in data
            assert "ai_message" in data
            assert "tool_calls" in data
            assert "tool_outputs" in data
            
            # Validate data types
            assert isinstance(data["success"], bool)
            assert isinstance(data["ai_message"], str)
            assert isinstance(data["tool_calls"], list)
            assert isinstance(data["tool_outputs"], list)
            
            # AI message should not be empty
            assert len(data["ai_message"].strip()) > 0
            
            print(f"Simple chat test passed: {data['ai_message'][:100]}...")

    @pytest.mark.asyncio
    async def test_chat_with_history(self):
        """Test chat endpoint with conversation history"""
        history = [
            {"type": "user", "content": "Hello"},
            {"type": "bot", "content": "Hi there! How can I help you today?"}
        ]
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
            response = await client.post("/chat", json={
                "message": "What tools do you have available?",
                "history": history
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["ai_message"]) > 0
            
            print("Chat with history test passed")

    @pytest.mark.asyncio
    async def test_chat_tool_usage(self):
        """Test chat endpoint that should trigger tool usage"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=90.0) as client:  # Even longer timeout for tool usage
            response = await client.post("/chat", json={
                "message": "Can you search for information about Python programming?",
                "history": []
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Should be successful even if tools aren't used
            assert "success" in data
            assert "ai_message" in data
            
            # If tools were used, validate structure
            if data.get("tool_calls"):
                assert isinstance(data["tool_calls"], list)
                for tool_call in data["tool_calls"]:
                    assert "name" in tool_call
                    assert isinstance(tool_call["name"], str)
            
            print(f"Tool usage test passed. Tools used: {len(data.get('tool_calls', []))}")

    @pytest.mark.asyncio
    async def test_chat_long_message(self):
        """Test chat endpoint with a longer message"""
        long_message = "Tell me about artificial intelligence and machine learning. " * 20
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
            response = await client.post("/chat", json={
                "message": long_message,
                "history": []
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["ai_message"]) > 0
            
            print("Long message test passed")

    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test how server handles edge cases"""
        # Test with None history (should be handled gracefully)
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = await client.post("/chat", json={
                "message": "Test message",
                "history": None
            })
            
            # Should either work (if None is converted to []) or return validation error
            assert response.status_code in [200, 422]
            
            print("Server error handling test passed")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test server can handle multiple concurrent requests"""
        async def make_request(client, message_num):
            try:
                response = await client.post("/chat", json={
                    "message": f"Test message {message_num}",
                    "history": []
                })
                return response.status_code, message_num
            except Exception as e:
                return e, message_num
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
            # Send 3 concurrent requests
            tasks = [make_request(client, i) for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least some should succeed
            success_count = 0
            for result in results:
                if not isinstance(result, Exception) and isinstance(result, tuple) and result[0] == 200:
                    success_count += 1
            
            assert success_count > 0, f"No successful concurrent requests. Results: {results}"
            
            print(f"Concurrent requests test passed: {success_count}/3 succeeded")


# Utility functions for debugging
@pytest.mark.asyncio
async def test_debug_server_status():
    """Debug function to check server status"""
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            response = await client.get("/health")
            print(f"Health Status: {response.status_code}")
            print(f"Health Data: {response.json()}")
            
            response = await client.get("/tools")
            print(f"Tools Status: {response.status_code}")
            print(f"Tools Data: {response.json()}")
            
    except Exception as e:
        print(f"Debug failed: {e}")


if __name__ == "__main__":
    # Run specific tests for debugging
    print("IMPORTANT: Make sure to start the server first with 'python app.py'")
    pytest.main([__file__, "-v", "-s"])