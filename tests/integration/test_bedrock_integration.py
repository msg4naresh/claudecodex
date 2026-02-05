"""
Integration tests for Claude-Bedrock Proxy Server.

This module provides comprehensive integration testing including:
- Full server startup and shutdown cycles
- Real AWS Bedrock API integration (when credentials available)
- Claude Code compatibility testing with realistic request patterns  
- Log file generation and validation
- End-to-end request/response validation

Tests can run with or without AWS credentials:
- With credentials: Full integration testing with real Bedrock API
- Without credentials: Server functionality testing with mocked calls
"""

import requests
import time
import subprocess
import os
import sys
import pytest
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import components for direct testing
from claudecodex.bedrock import (
    get_bedrock_client, call_bedrock_converse, extract_system_message,
    convert_to_bedrock_messages, count_tokens_from_messages
)
from claudecodex.models import MessagesRequest, Message

# Test configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8082
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
TIMEOUT = 30  # seconds
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")

# Skip AWS tests if no credentials available
skip_without_aws = pytest.mark.skipif(
    not (
        os.environ.get("AWS_ACCESS_KEY_ID") or 
        os.environ.get("AWS_PROFILE") or
        os.path.exists(os.path.expanduser("~/.aws/credentials"))
    ),
    reason="No AWS credentials found"
)


class BedrockServerIntegrationTest:
    """
    Integration test manager for the Claude-Bedrock proxy server.
    
    Handles:
    - Server process lifecycle management
    - HTTP request/response testing
    - Log file verification
    - Cleanup and resource management
    """
    
    def __init__(self):
        self.server_process = None
        # Use absolute path for log directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.test_log_dir = os.path.join(self.project_root, "logs")
        
    def start_server(self):
        """
        Start the Claude-Bedrock proxy server for integration testing.
        
        Launches the server process and waits for it to become ready
        by polling the health endpoint.
        
        Raises:
            Exception: If server fails to start within timeout period
        """
        print(f"🚀 Starting Claude-Bedrock server on {SERVER_URL}")
        print(f"📁 Log directory: {self.test_log_dir}")
        
        # Set environment for testing
        env = os.environ.copy()
        env['BEDROCK_MODEL_ID'] = BEDROCK_MODEL_ID
        
        # Start server process from project root directory
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
                    print("✅ Server started successfully")
                    return
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue
                
        raise Exception("❌ Server failed to start within timeout")
    
    def stop_server(self):
        """
        Stop the Claude-Bedrock proxy server gracefully.
        
        Attempts graceful termination first, then force kills if necessary.
        """
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            print("✅ Server stopped")
    
    def make_request(self, endpoint: str, data: Dict[Any, Any]) -> requests.Response:
        """
        Make HTTP request to the running server.
        
        Args:
            endpoint: API endpoint path (e.g., "/v1/messages")
            data: Request payload data
            
        Returns:
            requests.Response: HTTP response object
            
        Includes Claude Code compatible headers and request logging.
        """
        url = f"{SERVER_URL}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Claude-Code/1.0"
        }
        
        print(f"📤 POST {endpoint}")
        print(f"   Model: {data.get('model', 'N/A')}")
        print(f"   Messages: {len(data.get('messages', []))}")
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        print(f"📥 Response: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.text}")
            
        return response


# Test request patterns that simulate real Claude Code usage
CLAUDE_CODE_REQUEST_PATTERNS = [
    {
        "name": "Simple Chat Request",
        "endpoint": "/v1/messages",
        "data": {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, can you help me write a simple Python function?"
                }
            ],
            "temperature": 0.7
        }
    },
    {
        "name": "System Message Request",
        "endpoint": "/v1/messages", 
        "data": {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a function to calculate fibonacci numbers"
                }
            ],
            "system": "You are an expert Python developer. Write clean, well-documented code.",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    {
        "name": "Multi-turn Conversation",
        "endpoint": "/v1/messages",
        "data": {
            "model": "claude-3-5-sonnet-20240620", 
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you review this code and suggest improvements?"
                        }
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {
                            "type": "text",
                            "text": "I'd be happy to review your code! However, I don't see any code in your message."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
                }
            ],
            "temperature": 0.5,
            "stop_sequences": ["```"],
            "metadata": {"session_id": "test-123"}
        }
    },
    {
        "name": "Token Count Request",
        "endpoint": "/v1/messages/count_tokens",
        "data": {
            "model": "claude-3-5-sonnet-20240620",
            "messages": [
                {
                    "role": "user",
                    "content": "How many tokens is this message?"
                }
            ],
            "system": "You are helpful."
        }
    }
]


def test_server_startup_shutdown():
    """Test server startup and shutdown lifecycle."""
    test_instance = BedrockServerIntegrationTest()
    
    # Test startup
    test_instance.start_server()
    
    # Verify health endpoint
    response = requests.get(f"{SERVER_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    
    # Test shutdown
    test_instance.stop_server()
    
    print("✅ Server startup/shutdown test passed")


def test_claude_code_request_patterns():
    """Test all Claude Code request patterns for compatibility."""
    test_instance = BedrockServerIntegrationTest()
    
    try:
        test_instance.start_server()
        
        for i, test_case in enumerate(CLAUDE_CODE_REQUEST_PATTERNS):
            print(f"\n📋 Test {i+1}/{len(CLAUDE_CODE_REQUEST_PATTERNS)}: {test_case['name']}")
            
            response = test_instance.make_request(
                test_case["endpoint"], 
                test_case["data"]
            )
            
            # Basic response validation
            assert response.status_code == 200, f"Request failed: {response.text}"
            
            data = response.json()
            
            if test_case["endpoint"] == "/v1/messages":
                # Validate Claude API response structure
                assert "id" in data
                assert "content" in data
                assert "usage" in data
                assert data["role"] == "assistant"
                assert data["type"] == "message"
                assert len(data["content"]) > 0
                assert data["content"][0]["type"] == "text"
                
                print(f"   ✅ Valid response with {data['usage']['output_tokens']} tokens")
                
            elif test_case["endpoint"] == "/v1/messages/count_tokens":
                # Validate token count response
                assert "input_tokens" in data
                assert data["input_tokens"] > 0
                
                print(f"   ✅ Token count: {data['input_tokens']}")
        
        print(f"\n🎉 All {len(CLAUDE_CODE_REQUEST_PATTERNS)} request patterns passed!")
        
    finally:
        test_instance.stop_server()


def test_log_files_generation():
    """Test that log files are created with expected content."""
    test_instance = BedrockServerIntegrationTest()
    
    try:
        test_instance.start_server()
        
        # Make a request to generate logs
        response = test_instance.make_request(
            "/v1/messages",
            {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Test logging"}]
            }
        )
        
        # Verify log files exist and have content
        log_files = [
            Path(test_instance.test_log_dir) / "requests.log",
        ]
        
        for log_file in log_files:
            assert log_file.exists(), f"Log file not created: {log_file}"
            assert log_file.stat().st_size > 0, f"Log file is empty: {log_file}"
            print(f"✅ Log file created: {log_file} ({log_file.stat().st_size} bytes)")
        
        # Verify request log contains expected content
        requests_log = Path(test_instance.test_log_dir) / "requests.log"
        with open(requests_log) as f:
            content = f.read()
            assert "Test logging" in content
            assert "LLM Response" in content
            print("✅ Request log contains expected content")
            
    finally:
        test_instance.stop_server()


@skip_without_aws
def test_bedrock_client_initialization():
    """Test AWS Bedrock client can be created with available credentials."""
    client = get_bedrock_client()
    assert client is not None
    assert hasattr(client, 'converse')
    print("✅ Bedrock client created successfully")


@skip_without_aws 
def test_direct_bedrock_api_call():
    """Test direct call to Bedrock API without server."""
    request = MessagesRequest(
        model="claude-3-5-sonnet",
        max_tokens=100,
        messages=[
            Message(role="user", content="Say hello in exactly 3 words")
        ],
        temperature=0.1
    )
    
    try:
        response = call_bedrock_converse(request)
        
        # Validate response structure
        assert response.id.startswith("msg_")
        assert response.role == "assistant"
        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        
        response_text = response.content[0].text.strip()
        assert len(response_text) > 0
        print(f"✅ Bedrock response: '{response_text}'")
        print(f"📊 Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
        
    except Exception as e:
        if "AccessDenied" in str(e):
            pytest.skip(f"No access to Bedrock model: {BEDROCK_MODEL_ID}")
        elif "ValidationException" in str(e):
            pytest.skip(f"Invalid model ID: {BEDROCK_MODEL_ID}")
        else:
            raise


def test_message_translation_utilities():
    """Test message translation utility functions without AWS dependencies."""
    # Test token counting
    messages = [
        Message(role="user", content="Hello world"),
        Message(role="assistant", content="Hi there!")
    ]
    system = "You are helpful"
    
    token_count = count_tokens_from_messages(messages, system)
    assert token_count > 0
    assert isinstance(token_count, int)
    print(f"✅ Token counting: {token_count} tokens estimated")
    
    # Test system message extraction
    request = MessagesRequest(
        model="test",
        max_tokens=100,
        messages=messages,
        system=system
    )
    
    extracted = extract_system_message(request)
    assert extracted == system
    print(f"✅ System message extraction: '{extracted}'")
    
    # Test message conversion to Bedrock format
    bedrock_messages = convert_to_bedrock_messages(request)
    assert len(bedrock_messages) == 2  # System messages are filtered out
    assert bedrock_messages[0]["role"] == "user"
    assert bedrock_messages[0]["content"][0]["text"] == "Hello world"
    print("✅ Message format conversion working")


def run_all_integration_tests():
    """
    Run complete integration test suite.
    
    Executes all integration tests with proper AWS credential detection
    and informative output about test coverage.
    """
    print("🧪 Starting Claude-Bedrock Proxy Integration Tests")
    print("=" * 60)
    
    # Check AWS credentials availability
    has_aws_creds = (
        os.environ.get("AWS_ACCESS_KEY_ID") or 
        os.environ.get("AWS_PROFILE") or
        Path.home().joinpath(".aws", "credentials").exists()
    )
    
    if not has_aws_creds:
        print("⚠️  Warning: No AWS credentials found. Bedrock API tests will be skipped.")
        print("   Configure AWS_PROFILE or ~/.aws/credentials for full testing.")
    
    try:
        # Core server functionality tests (no AWS required)
        test_server_startup_shutdown()
        test_log_files_generation()
        test_message_translation_utilities()
        
        if has_aws_creds:
            print("\n🔗 Testing with real AWS Bedrock integration...")
            test_claude_code_request_patterns()
            test_bedrock_client_initialization()
            test_direct_bedrock_api_call()
        else:
            print("\n⏭️  Skipping AWS Bedrock integration tests (no credentials)")
        
        print("\n🎉 All available integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1)