import asyncio
import os
import logging
import signal
import sys
import warnings
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="FirecrawlAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://simple-agent-frontend-918169486800.us-central1.run.app",
        "http://127.0.0.1:5000"
    ],  # restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# Pydantic model for query input
class QueryRequest(BaseModel):
    input_text: str


# Global agent instance
agent_instance: Optional[FirecrawlAgent] = None


@app.on_event("startup")
async def startup_event():
    global agent_instance
    agent_instance = FirecrawlAgent()
    try:
        logger.info("Initializing FirecrawlAgent on startup...")
        await agent_instance.initialize()
        logger.info("FirecrawlAgent initialized successfully.")
    except (ConfigurationError, AgentError, ToolError) as e:
        logger.error(f"Agent initialization error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during agent initialization: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global agent_instance
    if agent_instance:
        logger.info("Cleaning up FirecrawlAgent on shutdown...")
        await agent_instance.cleanup()
        logger.info("Cleanup completed.")


@app.get("/")
async def read_root():
    return {"message": "Hello, world!"}

@app.get("/tools")
async def get_tools():
    global agent_instance
    if not agent_instance or not agent_instance._initialized:
        logger.error("Agent is not initialized or ready to serve tools")
        raise HTTPException(status_code=503, detail="Agent is not ready")

    try:
        # Return the tool names the agent loaded dynamically
        tools_list = [tool.name for tool in agent_instance.tools] if agent_instance.tools else []
        return {"tools": tools_list}
    except Exception as e:
        logger.error(f"Error fetching tools list: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tools")

@app.post("/query")
async def query_agent(request: QueryRequest):
    if not agent_instance or not agent_instance._initialized:
        logger.error("Agent is not initialized or ready")
        raise HTTPException(status_code=503, detail="Agent is not ready. Please try again later.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can scrape websites, crawl pages, "
                "and extract data using Firecrawl tools. Think step by step and use "
                "the appropriate tools to help the user."
            ),
        },
        {
            "role": "user",
            "content": request.input_text[:175000],
        },
    ]

    try:
        response = await asyncio.wait_for(
            agent_instance.agent.ainvoke({"messages": messages}),
            timeout=60.0  # adjust as necessary
        )

        ai_message_content = ""
        for msg in response.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                ai_message_content = msg.content
                break

        # Log tool usage and outputs, if any
        for msg in response.get("messages", []):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                logger.info(f"Agent used tools: {[tc.get('name', 'Unknown') for tc in msg.tool_calls]}")
            elif isinstance(msg, ToolMessage):
                logger.info(f"Tool output ({msg.name}): {str(msg.content)[:500]}")

        if not ai_message_content:
            ai_message_content = "(No textual reply from agent. Check tool outputs if any.)"

        return {"response": ai_message_content}

    except asyncio.TimeoutError:
        logger.error("Agent invocation timed out")
        raise HTTPException(status_code=504, detail="Agent request timed out.")
    except AgentError as e:
        logger.error(f"Agent error during invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during query processing: {e}")
        if "connection" in str(e).lower() or "closed" in str(e).lower():
            raise HTTPException(status_code=503, detail="Connection error to agent. Please retry.")
        raise HTTPException(status_code=500, detail="Internal server error.")


# Optional health check endpoint
@app.get("/health")
async def health_check():
    if agent_instance and agent_instance._initialized:
        return {"status": "ok"}
    else:
        raise HTTPException(status_code=503, detail="Agent is not initialized")


# Entry point for local dev/testing
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
