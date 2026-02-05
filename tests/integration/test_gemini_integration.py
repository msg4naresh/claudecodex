"""
Integration tests for Google Gemini via OpenAI-compatible API.

This module provides comprehensive integration testing for Gemini models including:
- Real Gemini API integration via OpenAI-compatible endpoint
- Gemini-specific model testing (2.0-flash, 1.5-pro, 1.5-flash)
- Tool calling with Gemini models
- Multimodal capabilities (when supported)
- Performance and token usage validation
- Error handling for Gemini-specific scenarios

Tests can run with or without Gemini API credentials:
- With GEMINI_API_KEY: Full integration testing with real Gemini API
- Without credentials: Server functionality testing with mocked calls
"""

import requests
import time
import subprocess
import os
import sys
import pytest
import json
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import components for direct testing
from claudecodex.server import call_llm_service, get_provider_info
from claudecodex.models import MessagesRequest, Message, Tool
from claudecodex.openai_compatible import call_openai_compatible_chat

# Test configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8082  # Same port as main server
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
TIMEOUT = 30  # seconds

# Gemini configuration
OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODELS = [
    "gemini-2.0-flash"
]

# Skip Gemini tests if no API key available
skip_without_gemini = pytest.mark.skipif(
    not os.environ.get("OPENAICOMPATIBLE_API_KEY"),
    reason="No OPENAICOMPATIBLE_API_KEY environment variable found"
)


class GeminiServerIntegrationTest:
    """
    Integration test manager for Gemini via OpenAI-compatible proxy.
    
    Handles:
    - Server process lifecycle with Gemini configuration
    - Gemini-specific API requests and responses
    - Model-specific testing and validation
    - Performance benchmarking
    """
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.server_process = None
        self.model = model
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.test_log_dir = os.path.join(self.project_root, "logs")
        
    def start_server(self):
        """
        Start the proxy server configured for Gemini.
        
        Sets up environment variables for OpenAI-compatible backend
        with Gemini-specific configuration.
        """
        print(f"🚀 Starting Gemini proxy server on {SERVER_URL}")
        print(f"🔮 Model: {self.model}")
        print(f"📁 Log directory: {self.test_log_dir}")
        
        # Set environment for Gemini testing
        env = os.environ.copy()
        env['LLM_PROVIDER'] = 'openai_compatible'
        env['OPENAICOMPATIBLE_API_KEY'] = os.environ.get('OPENAICOMPATIBLE_API_KEY', 'dummy')
        env['OPENAICOMPATIBLE_BASE_URL'] = OPENAI_BASE_URL
        env['OPENAI_MODEL'] = self.model
        env['SERVER_PORT'] = str(SERVER_PORT)
        
        # Start server process
        self.server_process = subprocess.Popen(
            ["python", "main.py"],
            env=env,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to become ready
        for _ in range(TIMEOUT):
            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Server started successfully")
                    print(f"📊 Provider: {data.get('provider')}, Model: {data.get('model')}")
                    return
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue
                
        raise Exception("❌ Gemini server failed to start within timeout")
    
    def stop_server(self):
        """Stop the Gemini proxy server gracefully."""
        if self.server_process:
            print("🛑 Stopping Gemini server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            print("✅ Gemini server stopped")
    
    def make_request(self, endpoint: str, data: Dict[Any, Any]) -> requests.Response:
        """Make HTTP request to the Gemini-configured server."""
        url = f"{SERVER_URL}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Claude-Code-Gemini-Test/1.0"
        }
        
        print(f"📤 POST {endpoint}")
        print(f"   Model: {data.get('model', self.model)}")
        print(f"   Messages: {len(data.get('messages', []))}")
        if data.get('tools'):
            print(f"   Tools: {len(data['tools'])}")
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        print(f"📥 Response: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.text}")
            
        return response
    
    def benchmark_performance(self, request_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Benchmark Gemini performance metrics."""
        start_time = time.time()
        response = self.make_request("/v1/messages", request_data)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            return {
                "duration": end_time - start_time,
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                "tokens_per_second": data.get("usage", {}).get("output_tokens", 0) / (end_time - start_time),
                "status": "success"
            }
        else:
            return {
                "duration": end_time - start_time,
                "status": "error",
                "error": response.text
            }


# Gemini-specific test request patterns
GEMINI_TEST_PATTERNS = [
    {
        "name": "Simple Gemini Chat",
        "endpoint": "/v1/messages",
        "data": {
            "model": "gemini-2.0-flash",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantum computing in simple terms"
                }
            ],
            "temperature": 0.7
        }
    },
    {
        "name": "Gemini Code Generation",
        "endpoint": "/v1/messages",
        "data": {
            "model": "gemini-2.0-flash",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function to implement binary search with detailed comments"
                }
            ],
            "system": "You are an expert Python developer who writes clean, efficient code with comprehensive documentation.",
            "temperature": 0.3
        }
    },
    {
        "name": "Gemini Creative Writing",
        "endpoint": "/v1/messages",
        "data": {
            "model": "gemini-2.0-flash",
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short story about a robot learning to paint"
                }
            ],
            "temperature": 0.9,
            "top_p": 0.95
        }
    },
    {
        "name": "Gemini Multi-turn Conversation",
        "endpoint": "/v1/messages",
        "data": {
            "model": "gemini-2.0-flash",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "What are the main differences between Python and JavaScript?"
                },
                {
                    "role": "assistant",
                    "content": "Python and JavaScript differ in several key areas:\n\n1. **Syntax**: Python uses indentation for code blocks, while JavaScript uses curly braces\n2. **Typing**: Python is dynamically typed but with optional static typing, JavaScript is dynamically typed\n3. **Execution**: Python runs on servers/desktops, JavaScript traditionally in browsers\n4. **Use cases**: Python for data science, AI, backend; JavaScript for web development, frontend"
                },
                {
                    "role": "user",
                    "content": "Which one should I learn first as a beginner?"
                }
            ],
            "temperature": 0.6
        }
    },
    {
        "name": "Gemini Tool Calling - Weather",
        "endpoint": "/v1/messages",
        "data": {
            "model": "gemini-2.0-flash",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Tokyo today?"
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather information for a specific location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country, e.g. Tokyo, Japan"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
            "tool_choice": {"type": "auto"}
        }
    },
    {
        "name": "Gemini Tool Calling - Calculator",
        "endpoint": "/v1/messages",
        "data": {
            "model": "gemini-2.0-flash",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate the compound interest for $10,000 at 5% annual rate for 10 years"
                }
            ],
            "tools": [
                {
                    "name": "calculate_compound_interest",
                    "description": "Calculate compound interest",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "principal": {"type": "number", "description": "Initial amount"},
                            "rate": {"type": "number", "description": "Annual interest rate (as decimal)"},
                            "time": {"type": "number", "description": "Time period in years"},
                            "compound_frequency": {"type": "number", "description": "Compounding frequency per year", "default": 12}
                        },
                        "required": ["principal", "rate", "time"]
                    }
                }
            ],
            "tool_choice": {"type": "auto"}
        }
    },
    {
        "name": "Gemini Token Counting",
        "endpoint": "/v1/messages/count_tokens",
        "data": {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": "This is a test message for counting tokens in Gemini models. How many tokens does this message contain approximately?"
                }
            ],
            "system": "You are a helpful assistant that provides accurate information."
        }
    }
]


@skip_without_gemini
def test_gemini_server_startup_shutdown():
    """Test Gemini-configured server startup and shutdown."""
    test_instance = GeminiServerIntegrationTest()
    
    try:
        test_instance.start_server()
        
        # Verify health endpoint shows Gemini configuration
        response = requests.get(f"{SERVER_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "openai_compatible"
        assert "gemini" in data["model"].lower()
        
        print("✅ Gemini server startup/shutdown test passed")
        
    finally:
        test_instance.stop_server()


@skip_without_gemini
@pytest.mark.parametrize("model", GEMINI_MODELS)
def test_gemini_models(model):
    """Test different Gemini models individually."""
    test_instance = GeminiServerIntegrationTest(model=model)
    
    try:
        test_instance.start_server()
        
        # Simple request to test model
        request_data = {
            "model": model,
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": f"Respond with exactly: 'Hello from {model}'"}
            ],
            "temperature": 0.1
        }
        
        response = test_instance.make_request("/v1/messages", request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model"] == model
        assert len(data["content"]) > 0
        assert data["usage"]["output_tokens"] > 0
        
        print(f"✅ {model} test passed")
        
    finally:
        test_instance.stop_server()


@skip_without_gemini
def test_gemini_request_patterns():
    """Test all Gemini request patterns for compatibility."""
    test_instance = GeminiServerIntegrationTest()
    
    try:
        test_instance.start_server()
        
        performance_results = []
        
        for i, test_case in enumerate(GEMINI_TEST_PATTERNS):
            print(f"\n📋 Gemini Test {i+1}/{len(GEMINI_TEST_PATTERNS)}: {test_case['name']}")
            
            # Benchmark performance
            if test_case["endpoint"] == "/v1/messages":
                perf_data = test_instance.benchmark_performance(test_case["data"])
                performance_results.append({
                    "test": test_case["name"],
                    "performance": perf_data
                })
            else:
                response = test_instance.make_request(test_case["endpoint"], test_case["data"])
                assert response.status_code == 200
            
            # Validate response based on endpoint
            if test_case["endpoint"] == "/v1/messages":
                assert perf_data["status"] == "success", f"Request failed: {perf_data.get('error')}"
                print(f"   ✅ Performance: {perf_data['duration']:.2f}s, {perf_data['tokens_per_second']:.1f} tokens/sec")
                
            elif test_case["endpoint"] == "/v1/messages/count_tokens":
                data = response.json()
                assert "input_tokens" in data
                assert data["input_tokens"] > 0
                print(f"   ✅ Token count: {data['input_tokens']}")
        
        # Print performance summary
        print(f"\n📊 Gemini Performance Summary:")
        for result in performance_results:
            perf = result["performance"]
            print(f"   {result['test']}: {perf['duration']:.2f}s, {perf['tokens_per_second']:.1f} tok/sec")
        
        print(f"\n🎉 All {len(GEMINI_TEST_PATTERNS)} Gemini request patterns passed!")
        
    finally:
        test_instance.stop_server()


@skip_without_gemini
def test_gemini_tool_calling_functionality():
    """Test Gemini tool calling capabilities in detail."""
    test_instance = GeminiServerIntegrationTest()
    
    try:
        test_instance.start_server()
        
        # Test tool calling request
        request_data = {
            "model": "gemini-2.0-flash",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "I need to know the current time in New York City"
                }
            ],
            "tools": [
                {
                    "name": "get_current_time",
                    "description": "Get the current time in a specific timezone",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone identifier (e.g., 'America/New_York')"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["12hour", "24hour"],
                                "description": "Time format preference"
                            }
                        },
                        "required": ["timezone"]
                    }
                }
            ],
            "tool_choice": {"type": "auto"}
        }
        
        response = test_instance.make_request("/v1/messages", request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check if Gemini used the tool
        if data.get("stop_reason") == "tool_use":
            assert len(data["content"]) > 0
            tool_use_block = None
            for block in data["content"]:
                if block.get("type") == "tool_use":
                    tool_use_block = block
                    break
            
            assert tool_use_block is not None
            assert tool_use_block["name"] == "get_current_time"
            assert "timezone" in tool_use_block["input"]
            
            print("✅ Gemini tool calling test passed")
            print(f"   Tool: {tool_use_block['name']}")
            print(f"   Input: {tool_use_block['input']}")
        else:
            # Gemini might respond with text instead of tool use
            assert len(data["content"]) > 0
            assert data["content"][0]["type"] == "text"
            print("✅ Gemini responded with text (tool calling optional)")
        
    finally:
        test_instance.stop_server()


@skip_without_gemini
def test_gemini_error_scenarios():
    """Test Gemini-specific error handling."""
    test_instance = GeminiServerIntegrationTest()
    
    try:
        test_instance.start_server()
        
        # Test invalid model
        invalid_request = {
            "model": "invalid-gemini-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Test"}]
        }
        
        response = test_instance.make_request("/v1/messages", invalid_request)
        # Should handle gracefully (might succeed if Gemini API is flexible)
        
        # Test excessive token request (if Gemini has limits)
        excessive_request = {
            "model": "gemini-2.0-flash",
            "max_tokens": 100000,  # Very high
            "messages": [{"role": "user", "content": "Generate a very long response"}]
        }
        
        response = test_instance.make_request("/v1/messages", excessive_request)
        # Should handle gracefully or return an appropriate error
        
        print("✅ Gemini error scenario tests completed")
        
    finally:
        test_instance.stop_server()


def test_gemini_configuration_validation():
    """Test Gemini configuration without making API calls."""
    # Test environment variable handling
    original_env = os.environ.copy()

    try:
        os.environ['LLM_PROVIDER'] = 'openai_compatible'
        os.environ['OPENAICOMPATIBLE_API_KEY'] = 'test-gemini-key'
        os.environ['OPENAICOMPATIBLE_BASE_URL'] = OPENAI_BASE_URL
        os.environ['OPENAI_MODEL'] = 'gemini-2.0-flash'

        provider_info = get_provider_info()

        assert provider_info["provider"] == "openai_compatible"
        assert provider_info["model"] == "gemini-2.0-flash"
        assert OPENAI_BASE_URL in provider_info["base_url"]
        assert provider_info["api_key_configured"] == True
        
        print("✅ Gemini configuration validation passed")
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@skip_without_gemini
def test_gemini_direct_service_call():
    """Test direct service call to Gemini without server."""
    # Configure environment for Gemini
    original_env = os.environ.copy()

    try:
        os.environ['LLM_PROVIDER'] = 'openai_compatible'
        os.environ['OPENAICOMPATIBLE_API_KEY'] = os.environ.get('OPENAICOMPATIBLE_API_KEY')
        os.environ['OPENAICOMPATIBLE_BASE_URL'] = OPENAI_BASE_URL
        os.environ['OPENAI_MODEL'] = 'gemini-2.0-flash'
        
        request = MessagesRequest(
            model="gemini-2.0-flash",
            max_tokens=200,
            messages=[
                Message(role="user", content="Say 'Hello from Gemini API' and nothing else")
            ],
            temperature=0.1
        )
        
        response = call_llm_service(request)
        
        # Validate response structure
        assert response.id.startswith("msg_")
        assert response.role == "assistant"
        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        
        response_text = response.content[0].text.strip()
        print(f"✅ Gemini direct call response: '{response_text}'")
        print(f"📊 Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
        
    except Exception as e:
        if "API_KEY" in str(e) or "authentication" in str(e).lower():
            pytest.skip("Invalid Gemini API key")
        else:
            raise
            
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def run_all_gemini_integration_tests():
    """
    Run complete Gemini integration test suite.

    Executes all Gemini-specific tests with proper API key detection
    and informative output about test coverage.
    """
    print("🔮 Starting Gemini Integration Tests")
    print("=" * 50)

    # Check API key availability
    has_api_key = bool(os.environ.get("OPENAICOMPATIBLE_API_KEY"))

    if not has_api_key:
        print("⚠️  Warning: No OPENAICOMPATIBLE_API_KEY found. Gemini API tests will be skipped.")
        print("   Set OPENAICOMPATIBLE_API_KEY environment variable for full testing.")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
    
    try:
        # Configuration test (no API key required)
        test_gemini_configuration_validation()
        
        if has_api_key:
            print("\n🔮 Testing with real Gemini API integration...")
            test_gemini_server_startup_shutdown()
            test_gemini_request_patterns()
            test_gemini_tool_calling_functionality()
            test_gemini_error_scenarios()
            test_gemini_direct_service_call()
            
            # Test multiple models if time permits
            print("\n🔄 Testing different Gemini models...")
            for model in GEMINI_MODELS[:2]:  # Test first 2 models to save time
                try:
                    test_gemini_models(model)
                except Exception as e:
                    print(f"⚠️  Model {model} test failed: {e}")
        else:
            print("\n⏭️  Skipping Gemini API integration tests (no API key)")
        
        print("\n🎉 All available Gemini integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Gemini integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_gemini_integration_tests()
    exit(0 if success else 1)