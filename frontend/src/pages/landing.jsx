import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Loader,
  Search,
  Globe,
  FileText,
  Database,
  Brain,
  Zap,
  AlertCircle,
  ChevronDown, // Import ChevronDown icon
} from "lucide-react";
// Import Card components from shadcn/ui
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
// Import Collapsible components
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

function LandingPage({ onNavigate }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "bot",
      content:
        "Hello! I'm your AI agent powered by Gemini LLM and Firecrawl tools. I can help you search the web, scrape content, extract data, and provide intelligent responses. What would you like to know?",
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [availableTools, setAvailableTools] = useState([]);
  const [toolsLoading, setToolsLoading] = useState(true);
  const [toolsError, setToolsError] = useState(null);
  const [isToolsCollapsibleOpen, setIsToolsCollapsibleOpen] = useState(false); // State for collapsible

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Backend URL - automatically detects environment
  const getBackendURL = () => {
    // Check if we're in development (localhost)
    if (
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1"
    ) {
      return "http://localhost:5000";
    }
    // Production backend URL
    return "https://simple-agent-backend-918169486800.us-central1.run.app";
  };

  const BACKEND_URL = getBackendURL();

  // Function to format AI responses for better readability
  const formatAIResponse = (content) => {
    // If content is already well-formatted (contains bullet points, line breaks, etc.), return as is
    if (
      content.includes("•") ||
      content.includes("**") ||
      content.includes("\n\n")
    ) {
      return content;
    }

    // For lengthy paragraphs, try to break them up
    const sentences = content
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 0);

    // If it's a long paragraph with multiple sentences, format it better
    if (sentences.length > 3 && content.length > 300) {
      // Look for patterns that suggest lists or multiple items
      const listPatterns = [
        /\*\*([^*]+)\*\*/g, // Bold items
        /(\w+\.com|\w+\.org|\w+\.ai)/g, // Website mentions
        /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[:–-]/g, // Title patterns
      ];

      let formatted = content;

      // Try to identify and format website/platform mentions
      const websiteMatches = content.match(
        /(\w+\.com|\w+\.org|\w+\.ai|\w+Face|\w+Net|[A-Z][a-z]+[A-Z][a-z]+)/g
      );
      if (websiteMatches && websiteMatches.length > 2) {
        // This looks like a list of platforms/websites
        let formattedContent = "I found several great options:\n\n";

        websiteMatches.forEach((site, index) => {
          // Extract the sentence or context around this site
          const siteRegex = new RegExp(
            `([^.]*${site.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[^.]*\\.?)`,
            "i"
          );
          const match = content.match(siteRegex);
          if (match) {
            formattedContent += `• **${site}** - ${match[1]
              .replace(site, "")
              .trim()}\n`;
          }
        });

        formattedContent +=
          "\nWould you like me to provide more details about any of these options?";
        return formattedContent;
      }

      // If no specific patterns found, just break into shorter paragraphs
      const midPoint = Math.floor(sentences.length / 2);
      const firstHalf = sentences.slice(0, midPoint).join(". ") + ".";
      const secondHalf = sentences.slice(midPoint).join(". ") + ".";

      return `${firstHalf}\n\n${secondHalf}`;
    }

    return content;
  };

  // Fetch available tools from the backend
  const fetchAvailableTools = async () => {
    try {
      setToolsLoading(true);
      setToolsError(null);

      const response = await fetch(`${BACKEND_URL}/tools`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        // Add timeout for production
        signal: AbortSignal.timeout(10000), // 10 second timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.tools && Array.isArray(data.tools)) {
        setAvailableTools(data.tools);
      } else {
        throw new Error("Invalid tools data format");
      }
    } catch (error) {
      console.error("Error fetching available tools:", error);
      setToolsError(error.message);
      // Fallback to hardcoded tools if fetch fails
      setAvailableTools([
        "firecrawl_scrape",
        "firecrawl_map",
        "firecrawl_crawl",
        "firecrawl_check_crawl_status",
        "firecrawl_search",
        "firecrawl_extract",
        "firecrawl_deep_research",
        "firecrawl_generate_llmstxt",
      ]);
    } finally {
      setToolsLoading(false);
    }
  };

  // Fetch tools on component mount
  useEffect(() => {
    fetchAvailableTools();
  }, []);

  // Scrolls to the bottom of the messages container
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Effect to scroll to bottom whenever messages update
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handles the submission of a user message and communicates with the backend
  const handleSubmit = async () => {
  if (!inputValue.trim() || isLoading) return;

  const userMessage = {
    id: Date.now(),
    type: "user",
    content: inputValue,
    timestamp: new Date().toLocaleTimeString(),
  };

  setMessages((prev) => [...prev, userMessage]);
  setInputValue("");
  setIsLoading(true);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout

    const response = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input_text: userMessage.content,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      if (response.status === 503) {
        throw new Error("Service is starting up. Please wait a moment and try again.");
      } else if (response.status >= 500) {
        throw new Error("Server error. Please try again later.");
      } else if (response.status === 404) {
        throw new Error("Service not found. Please check the backend URL.");
      } else {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    }

    const data = await response.json();

    const botMessageContent = data.response || "(No direct textual response)";

    // Format AI response as in your existing function
    const formattedContent = formatAIResponse(botMessageContent);

    const botResponse = {
      id: Date.now() + 1,
      type: "bot",
      content: formattedContent,
      timestamp: new Date().toLocaleTimeString(),
      toolsUsed: [],  // Your backend currently does not return tool calls; update if backend supports it
    };

    setMessages((prev) => [...prev, botResponse]);
  } catch (error) {
    console.error("Error sending message to backend:", error);

    let errorMessage = "⚠️ **Connection Error**\n\n";

    if (error.name === "AbortError") {
      errorMessage +=
        "Request timed out. The server might be busy.\n• Try a simpler question\n• Wait a moment and try again\n• Check your internet connection";
    } else if (error.message.includes("Service is starting up")) {
      errorMessage +=
        "The AI service is starting up.\n• Please wait 30-60 seconds\n• Try your request again\n• The service should be ready shortly";
    } else if (
      error.message.includes("Failed to fetch") ||
      error.message.includes("NetworkError")
    ) {
      errorMessage +=
        "Network connection issue.\n• Check your internet connection\n• The server might be temporarily unavailable\n• Try again in a few moments";
    } else {
      errorMessage += `Sorry, I couldn't process your request:\n• ${error.message}\n• Please try again in a moment`;
    }

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now() + 1,
        type: "bot",
        content: errorMessage,
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
  } finally {
    setIsLoading(false);
  }
};

  // Returns the appropriate Lucide icon for a given tool
  const getToolIcon = (tool) => {
    if (tool.includes("search")) return <Search className="w-3 h-3" />;
    if (tool.includes("scrape")) return <Globe className="w-3 h-3" />;
    if (tool.includes("extract")) return <FileText className="w-3 h-3" />;
    if (tool.includes("crawl")) return <Database className="w-3 h-3" />;
    return <Zap className="w-3 h-3" />;
  };

  // Component to render formatted message content
  const MessageContent = ({ content }) => {
    // Split content by line breaks and render with proper formatting
    const lines = content.split("\n");

    return (
      <div className="space-y-2">
        {lines.map((line, index) => {
          // Skip empty lines
          if (!line.trim()) return <div key={index} className="h-2" />;

          // Handle bullet points
          if (line.trim().startsWith("•")) {
            return (
              <div key={index} className="flex items-start gap-2">
                <span className="text-blue-500 font-bold">•</span>
                <span
                  className="flex-1"
                  dangerouslySetInnerHTML={{
                    __html: line
                      .replace("•", "")
                      .trim()
                      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"),
                  }}
                />
              </div>
            );
          }

          // Handle bold headings
          if (line.includes("**") && line.trim().endsWith("**")) {
            return (
              <div
                key={index}
                className="font-bold text-lg mt-3 mb-1"
                dangerouslySetInnerHTML={{
                  __html: line.replace(/\*\*(.*?)\*\*/g, "$1"),
                }}
              />
            );
          }

          // Regular line with bold formatting
          return (
            <p
              key={index}
              className="leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"),
              }}
            />
          );
        })}
      </div>
    );
  };

  // Render tools section with loading and error states
  const renderToolsSection = () => {
    return (
      <Collapsible
        open={isToolsCollapsibleOpen}
        onOpenChange={setIsToolsCollapsibleOpen}
        className="bg-black/20 backdrop-blur-lg border-b border-purple-500/10 p-3"
      >
        <CollapsibleTrigger asChild>
          <div className="flex items-center justify-between cursor-pointer py-1">
            <div className="flex items-center gap-2">
              <p className="text-sm text-purple-300">Available Tools:</p>
              {toolsLoading && (
                <div className="flex items-center gap-2 ml-2">
                  <Loader className="w-4 h-4 animate-spin text-purple-400" />
                  <span className="text-xs text-purple-400">Connecting...</span>
                </div>
              )}
              {toolsError && !toolsLoading && (
                <div className="flex items-center gap-2 ml-2">
                  <AlertCircle className="w-4 h-4 text-amber-400" />
                  <span className="text-xs text-amber-400">
                    Error (using fallback)
                  </span>
                </div>
              )}
            </div>
            <ChevronDown
              className={`w-5 h-5 text-purple-300 transition-transform duration-200 ${
                isToolsCollapsibleOpen ? "rotate-180" : ""
              }`}
            />
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="pt-2">
            {toolsLoading ? (
              <p className="text-xs text-purple-400">
                Loading tools from backend...
              </p>
            ) : toolsError ? (
              <div className="flex flex-wrap gap-2">
                {availableTools.map((tool, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-amber-500/20 text-amber-200 rounded text-xs font-mono flex items-center gap-1"
                  >
                    {getToolIcon(tool)}
                    {tool}
                  </span>
                ))}
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {availableTools.map((tool, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-purple-500/20 text-purple-200 rounded text-xs font-mono flex items-center gap-1"
                  >
                    {getToolIcon(tool)}
                    {tool}
                  </span>
                ))}
              </div>
            )}
            <button
              onClick={fetchAvailableTools}
              className="text-xs text-purple-400 hover:text-purple-300 transition-colors mt-2"
              title="Refresh tools"
            >
              🔄 Refresh Tools
            </button>
          </div>
        </CollapsibleContent>
      </Collapsible>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4 sm:p-2 md:p-4 font-sans">
      {" "}
      {/* Adjusted padding */}
      {/* Main Chat Container Card */}
      <Card className="w-full sm:w-[95%] md:w-[85%] lg:w-[80%] xl:w-[75%] max-w-6xl h-[90vh] flex flex-col rounded-xl overflow-hidden bg-black/40 backdrop-blur-xl border border-purple-500/20 shadow-lg shadow-purple-900/50">
        {" "}
        {/* Adjusted width */}
        {/* Header - Now part of the main Card */}
        <CardHeader className="bg-black/30 border-b border-purple-500/20 p-4 flex flex-row items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <CardTitle className="text-xl font-bold text-white m-0">
                AI Agent
              </CardTitle>
              <CardDescription className="text-sm text-purple-300 m-0">
                Powered by Gemini LLM & Firecrawl
              </CardDescription>
            </div>
          </div>
          <div className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium">
            {window.location.hostname === "localhost" ? "Local" : "Online"}
          </div>
        </CardHeader>
        {/* Available Tools - Dynamic loading */}
        {renderToolsSection()}
        {/* Messages */}
        <CardContent className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 ${
                message.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`flex gap-3 max-w-[80%] ${
                  message.type === "user" ? "flex-row-reverse" : "flex-row"
                }`}
              >
                {/* Avatar */}
                <div
                  className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                    message.type === "user"
                      ? "bg-gradient-to-r from-blue-500 to-cyan-500"
                      : "bg-gradient-to-r from-purple-500 to-pink-500"
                  )}
                >
                  {message.type === "user" ? (
                    <User className="w-5 h-5 text-white" />
                  ) : (
                    <Bot className="w-5 h-5 text-white" />
                  )}
                </div>

                {/* Message Card */}
                <Card
                  className={cn(
                    "rounded-2xl",
                    message.type === "user"
                      ? "bg-gradient-to-r from-blue-500 to-cyan-500 text-white border-none shadow-md"
                      : "bg-background text-card-foreground border border-purple-500/20 shadow-md"
                  )}
                >
                  <CardContent className="p-4">
                    {message.type === "bot" ? (
                      <MessageContent content={message.content} />
                    ) : (
                      <p className="text-sm leading-relaxed m-0">
                        {message.content}
                      </p>
                    )}

                    {message.toolsUsed && message.toolsUsed.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-purple-500/20">
                        <p className="text-xs text-gray-800 dark:text-gray-200 font-semibold mb-2">
                          Tools Used:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {message.toolsUsed.map((tool, index) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-gradient-to-r from-emerald-500/90 to-teal-500/90 text-white rounded-md text-xs font-mono flex items-center gap-1 shadow-sm border border-emerald-400/30"
                            >
                              {getToolIcon(tool)}
                              <span className="font-medium">{tool}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="mt-2 text-xs opacity-70">
                      {message.timestamp}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="flex gap-3 max-w-[80%]">
                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <Card className="bg-background text-card-foreground border border-purple-500/20 shadow-md">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 text-purple-300">
                      <Loader className="w-4 h-4 animate-spin" />
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </CardContent>
        {/* Input */}
        <div className="bg-black/30 backdrop-blur-xl border-t border-purple-500/20 p-4">
          <div className="flex gap-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="Ask me anything... I can search, scrape, extract data, and more!"
              className="flex-1 px-4 py-3 bg-black/40 backdrop-blur-lg border border-purple-500/30 rounded-2xl text-white text-sm outline-none placeholder:text-purple-300 focus:border-purple-500 focus:shadow-lg focus:shadow-purple-500/20 disabled:opacity-50 transition-all duration-200"
              disabled={isLoading}
            />
            <button
              onClick={handleSubmit}
              disabled={!inputValue.trim() || isLoading}
              className={`px-6 py-3 rounded-2xl flex items-center gap-2 font-medium text-white transition-all duration-200 ${
                isLoading || !inputValue.trim()
                  ? "bg-gray-500/50 cursor-not-allowed"
                  : "bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 cursor-pointer shadow-lg hover:shadow-purple-500/25"
              }`}
            >
              {isLoading ? (
                <Loader className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              {isLoading ? "Processing..." : "Send"}
            </button>
          </div>

          <div className="mt-2 text-xs text-purple-400 text-center">
            Powered by Gemini LLM • Firecrawl Tools • Real-time Web Intelligence
            {window.location.hostname !== "localhost" && (
              <span className="ml-2">• Deployed on Render</span>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}

export default LandingPage;
