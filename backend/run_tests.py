#!/usr/bin/env python3
"""
Test Runner Script for AI Agent Backend

This script helps you run the tests properly by checking if the server is running
and providing helpful instructions.
"""

import subprocess
import sys
import time
import httpx
import asyncio
import os


async def check_server_status():
    """Check if the server is running"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception:
        return False


def run_server_in_background():
    """Try to start the server in the background"""
    try:
        # Check if app.py exists
        if not os.path.exists("app.py"):
            print("❌ app.py not found in current directory")
            return None
        
        print("🚀 Starting server in background...")
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None


async def wait_for_server(max_wait=30):
    """Wait for server to become available"""
    print("⏳ Waiting for server to start...")
    
    for i in range(max_wait):
        if await check_server_status():
            print("✅ Server is ready!")
            return True
        
        if i % 5 == 0:
            print(f"   Still waiting... ({i}/{max_wait}s)")
        
        await asyncio.sleep(1)
    
    return False


def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    
    # Run pytest with proper configuration
    cmd = [
        sys.executable, "-m", "pytest", 
        "test_app.py", 
        "-v", 
        "-s",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


async def main():
    """Main test runner"""
    print("🔧 AI Agent Backend Test Runner")
    print("=" * 40)
    
    # Check if server is already running
    server_running = await check_server_status()
    server_process = None
    
    if not server_running:
        print("📡 Server not detected on localhost:8000")
        
        # Ask user what to do
        choice = input("\nOptions:\n1. Start server automatically\n2. I'll start it manually\nChoose (1 or 2): ").strip()
        
        if choice == "1":
            server_process = run_server_in_background()
            if server_process:
                # Wait for server to start
                if not await wait_for_server():
                    print("❌ Server failed to start within 30 seconds")
                    if server_process:
                        server_process.terminate()
                    return False
            else:
                return False
        else:
            print("\n📋 Manual setup required:")
            print("1. Open a new terminal")
            print("2. Run: python app.py")
            print("3. Wait for 'AI Agent is ready to help!' message")
            print("4. Come back here and press Enter")
            input("\nPress Enter when server is running...")
            
            if not await check_server_status():
                print("❌ Server still not accessible")
                return False
    else:
        print("✅ Server is already running!")
    
    # Run the tests
    try:
        success = run_tests()
        
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed - check output above")
        
        return success
        
    finally:
        # Cleanup background server if we started it
        if server_process:
            print("\n🛑 Stopping background server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚡ Test run cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)