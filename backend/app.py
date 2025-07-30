import asyncio
import os
import logging
import signal
import sys
import warnings
from typing import Optional
from dotenv import load_dotenv

# Suppress asyncio resource warnings on Windows
if sys.platform == "win32":
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class AgentError(Exception):
    """Custom exception for agent-related errors"""
    pass


class ToolError(Exception):
    """Custom exception for tool-related errors"""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class FirecrawlAgent:
    def __init__(self):
        self.model = None
        self.session = None
        self.stdio_context = None
        self.agent = None
        self.tools = None
        self._shutdown_event = asyncio.Event()
        self._initialized = False

    async def initialize(self):
        """Initialize the agent with proper error handling"""
        try:
            # Validate environment variables
            if not os.getenv("FIRECRAWL_API_KEY"):
                raise ConfigurationError("FIRECRAWL_API_KEY environment variable is required")

            # Initialize model with error handling
            try:
                self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
                logger.info("Model initialized successfully")
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize Google Generative AI model: {e}")

            # Setup server parameters
            server_params = StdioServerParameters(
                command="npx",
                env={
                    "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
                },
                args=["firecrawl-mcp"],
            )

            # Initialize MCP client using proper async context manager
            try:
                self.stdio_context = stdio_client(server_params)
                read, write = await self.stdio_context.__aenter__()
                logger.info("MCP client initialized successfully")
            except Exception as e:
                raise AgentError(f"Failed to initialize MCP client: {e}")

            # Initialize session
            try:
                self.session = ClientSession(read, write)
                await self.session.__aenter__()
                await self.session.initialize()
                logger.info("MCP session initialized successfully")
            except Exception as e:
                raise AgentError(f"Failed to initialize MCP session: {e}")

            # Load tools
            try:
                self.tools = await load_mcp_tools(self.session)
                logger.info(f"Loaded {len(self.tools)} tools successfully")
            except Exception as e:
                raise ToolError(f"Failed to load MCP tools: {e}")

            # Create agent
            try:
                self.agent = create_react_agent(self.model, self.tools)
                logger.info("Agent created successfully")
            except Exception as e:
                raise AgentError(f"Failed to create agent: {e}")

            self._initialized = True

        except Exception as e:
            await self.cleanup()
            raise

    async def cleanup(self):
        """Properly cleanup resources"""
        if not self._initialized:
            return

        logger.info("Starting cleanup process...")

        # Clean up session first
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
                logger.info("Session closed")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self.session = None

        # Clean up stdio context
        if self.stdio_context:
            try:
                await self.stdio_context.__aexit__(None, None, None)
                logger.info("MCP client connections closed")
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
            finally:
                self.stdio_context = None

        # Small delay to allow subprocess cleanup
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

        self._initialized = False

    async def get_user_input(self) -> Optional[str]:
        """Get user input with proper error handling"""
        try:
            # Use asyncio to handle input with timeout
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input, "\nYou: ")
            return user_input.strip()
        except EOFError:
            logger.info("EOF encountered, user wants to exit")
            return None
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, user wants to exit")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during input: {e}")
            return None

    async def process_message(self, user_input: str, messages: list) -> bool:
        """Process a user message with comprehensive error handling"""
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            return False

        # Truncate input if too long
        truncated_input = user_input[:175000]
        if len(user_input) > 175000:
            logger.warning(f"Input truncated from {len(user_input)} to 175000 characters")

        messages.append({"role": "user", "content": truncated_input})

        try:
            # Check if agent is still initialized
            if not self._initialized or not self.agent:
                raise AgentError("Agent is not properly initialized")

            # Invoke agent with timeout
            agent_response = await asyncio.wait_for(
                self.agent.ainvoke({"messages": messages}),
                timeout=60.0
            )

            # Process response
            ai_message_content = ""
            for msg in agent_response["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    print("\n--- LLM USED A TOOL! ---")
                    for tool_call in msg.tool_calls:
                        print(f"  Tool: {tool_call.get('name', 'Unknown')}")
                    print("------------------------")
                elif isinstance(msg, ToolMessage):
                    print(f"\n--- TOOL OUTPUT ({msg.name}): ---")
                    # Safely display tool output
                    content = str(msg.content)[:500]
                    print(content + ("..." if len(str(msg.content)) > 500 else ""))
                    print("---------------------------")
                elif isinstance(msg, AIMessage) and msg.content:
                    ai_message_content = msg.content

            if ai_message_content:
                print("\nAgent:", ai_message_content)
            else:
                print("\nAgent: (Processing completed - check tool outputs above)")

        except asyncio.TimeoutError:
            logger.error("Agent invocation timed out")
            print("\nError: Request timed out. Please try a simpler query or check your connection.")
        except AgentError as e:
            logger.error(f"Agent error: {e}")
            print(f"\nAgent Error: {e}")
            return False  # Stop processing if agent is broken
        except Exception as e:
            logger.error(f"Error during agent invocation: {e}")
            print(f"\nError: Something went wrong while processing your request. Please try again.")
            # Check if this is a connection error that requires restart
            if "connection" in str(e).lower() or "closed" in str(e).lower():
                logger.error("Connection error detected, stopping agent")
                return False

        return True

    async def run(self):
        """Main run loop with error handling"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user. If you encounter any errors, explain them clearly to the user.",
            }
        ]

        print("Available Tools:", *[tool.name for tool in self.tools])
        print("-" * 60)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Press Ctrl+C to force quit")

        try:
            while not self._shutdown_event.is_set():
                user_input = await self.get_user_input()

                if user_input is None:  # EOF or error
                    break

                if not await self.process_message(user_input, messages):
                    break

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            print("\nReceived interrupt signal. Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            print(f"\nUnexpected error: {e}")
        finally:
            await self.cleanup()


async def main():
    """Main function with comprehensive error handling"""
    agent = FirecrawlAgent()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown")
        agent._shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await agent.initialize()
        await agent.run()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nConfiguration Error: {e}")
        print("Please check your environment variables and configuration.")
        sys.exit(1)
    except AgentError as e:
        logger.error(f"Agent error: {e}")
        print(f"\nAgent Error: {e}")
        print("Please check your network connection and try again.")
        sys.exit(1)
    except ToolError as e:
        logger.error(f"Tool error: {e}")
        print(f"\nTool Error: {e}")
        print("Please check your Firecrawl API key and permissions.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nUnexpected Error: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Failed to start: {e}")
        sys.exit(1)