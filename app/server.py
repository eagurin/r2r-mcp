#!/usr/bin/env python3
"""
R2R Retrieval System - Model Context Protocol (MCP) Server

This MCP server enhances Claude with retrieval and search capabilities by providing
tools to search through knowledge bases, perform vector searches, graph searches,
web searches, and document searches.

Installation:
    # Install dependencies with uv
    uv pip install mcp r2r loguru

    # Local R2R API:
    mcp install app/server.py -v R2R_API_URL=http://localhost:7272

    # Cloud R2R API:
    mcp install app/server.py -v R2R_API_KEY=your_api_key_here

Usage with Claude:
    Once installed, Claude can use the R2R tools when appropriate or when explicitly requested.
"""

import warnings

# Фильтрация предупреждений от Pydantic
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="pydantic"
)

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from r2r import R2RAsyncClient

# Import FastMCP for server implementation
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError("MCP is not installed. Please run `uv pip install mcp`")

# Import R2R client
try:
    from r2r import R2RAsyncClient, R2RClient
except ImportError:
    raise ImportError(
        "R2R client is not installed. Please run `uv pip install r2r`"
    )

# Создаем директорию logs, если она не существует
logs_dir = Path("../logs")
logs_dir.mkdir(exist_ok=True)

# Import Loguru for better logging
try:
    from loguru import logger

    # Имя лог-файла с timestamp для уникальности
    log_filename = (
        logs_dir / f"r2r_mcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        log_filename,
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        compression="zip",
    )
    logger.info(f"Логирование настроено. Лог-файл: {log_filename}")
except ImportError:
    # Fallback to basic logging if loguru is not available
    import logging

    # Настраиваем базовое логирование с сохранением в файл
    log_filename = str(
        logs_dir / f"r2r_mcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logger = logging.getLogger("r2r_mcp")
    logger.info(f"Логирование настроено. Лог-файл: {log_filename}")


def id_to_shorthand(id: str) -> str:
    """Convert a long ID to a shorter version for display."""
    if not id:
        return "unknown"
    return str(id)[:7]


def format_search_results_for_llm(results) -> str:
    """
    Format search results in a structured way for LLM consumption.

    Args:
        results: The search results from R2R

    Returns:
        A formatted string with search results
    """
    lines = []

    # 1) Chunk search (vector search)
    if (
        hasattr(results, "chunk_search_results")
        and results.chunk_search_results
    ):
        lines.append("## Vector Search Results:")
        for i, c in enumerate(
            results.chunk_search_results[:5], 1
        ):  # Limit to top 5
            chunk_id = getattr(c, "id", f"chunk_{i}")
            chunk_text = getattr(c, "text", "")
            chunk_score = getattr(c, "score", "N/A")

            lines.append(
                f"Source ID [{id_to_shorthand(chunk_id)}] (Score: {chunk_score}):"
            )
            # Truncate long text for readability
            if len(chunk_text) > 500:
                lines.append(f"{chunk_text[:500]}...")
            else:
                lines.append(chunk_text)
            lines.append("")  # Empty line for separation

    # 2) Graph search
    if (
        hasattr(results, "graph_search_results")
        and results.graph_search_results
    ):
        lines.append("## Graph Search Results:")
        for i, g in enumerate(
            results.graph_search_results[:5], 1
        ):  # Limit to top 5
            graph_id = getattr(g, "id", f"graph_{i}")
            lines.append(f"Source ID [{id_to_shorthand(graph_id)}]:")

            # Handle different types of graph content
            content = getattr(g, "content", None)
            if content:
                if hasattr(content, "summary"):
                    lines.append(
                        f"Community Name: {getattr(content, 'name', 'Unnamed')}"
                    )
                    lines.append(f"ID: {getattr(content, 'id', 'No ID')}")
                    lines.append(
                        f"Summary: {getattr(content, 'summary', 'No summary')}"
                    )
                elif hasattr(content, "name") and hasattr(
                    content, "description"
                ):
                    lines.append(f"Entity Name: {content.name}")
                    lines.append(f"Description: {content.description}")
                elif (
                    hasattr(content, "subject")
                    and hasattr(content, "predicate")
                    and hasattr(content, "object")
                ):
                    lines.append(
                        f"Relationship: {content.subject} - {content.predicate} - {content.object}"
                    )
                else:
                    # Fallback for unknown content structure
                    lines.append(f"Content: {str(content)}")
            lines.append("")  # Empty line for separation

    # 3) Web search
    if hasattr(results, "web_search_results") and results.web_search_results:
        lines.append("## Web Search Results:")
        for i, w in enumerate(
            results.web_search_results[:5], 1
        ):  # Limit to top 5
            web_id = getattr(w, "id", f"web_{i}")
            title = getattr(w, "title", "Untitled")
            link = getattr(w, "link", "No link")
            snippet = getattr(w, "snippet", "No snippet")

            lines.append(f"Source ID [{id_to_shorthand(web_id)}]:")
            lines.append(f"Title: {title}")
            lines.append(f"Link: {link}")
            lines.append(f"Snippet: {snippet}")
            lines.append("")  # Empty line for separation

    # 4) Local context docs
    if (
        hasattr(results, "document_search_results")
        and results.document_search_results
    ):
        lines.append("## Local Context Documents:")
        for i, doc_result in enumerate(
            results.document_search_results[:3], 1
        ):  # Limit to top 3
            doc_title = getattr(doc_result, "title", "Untitled Document")
            doc_id = getattr(doc_result, "id", f"doc_{i}")
            summary = getattr(doc_result, "summary", "No summary available")

            lines.append(f"Document {i}:")
            lines.append(f"Full Document ID: {doc_id}")
            lines.append(f"Shortened Document ID: {id_to_shorthand(doc_id)}")
            lines.append(f"Document Title: {doc_title}")
            if summary:
                lines.append(f"Summary: {summary}")

            # Handle document chunks
            chunks = getattr(doc_result, "chunks", [])
            if chunks:
                for j, chunk in enumerate(
                    chunks[:2], 1
                ):  # Limit to 2 chunks per document
                    chunk_id = chunk.get("id", f"chunk_{j}")
                    chunk_text = chunk.get("text", "No text")

                    lines.append(f"\nChunk ID {id_to_shorthand(chunk_id)}:")
                    # Truncate long text for readability
                    if len(chunk_text) > 300:
                        lines.append(f"{chunk_text[:300]}...")
                    else:
                        lines.append(chunk_text)
            lines.append("")  # Empty line for separation

    # If no results were found
    if not lines:
        return "No search results found for the given query."

    result = "\n".join(lines)
    return result


def format_rag_response(rag_response: Union[Dict[str, Any], Any]) -> str:
    """Format RAG response for LLM consumption."""
    if not rag_response:
        return "No answer was generated."

    # Extract answer
    answer = None
    citations = []

    # Try to extract answer from different object structures
    if isinstance(rag_response, dict):
        answer = rag_response.get("answer")
        citations = rag_response.get("citations", [])
    else:
        if hasattr(rag_response, "results") and rag_response.results:
            results = rag_response.results
            if hasattr(results, "generated_answer"):
                answer = results.generated_answer
            elif hasattr(results, "answer"):
                answer = results.answer

            if hasattr(results, "citations"):
                citations = results.citations
        elif hasattr(rag_response, "generated_answer"):
            answer = rag_response.generated_answer
        elif hasattr(rag_response, "answer"):
            answer = rag_response.answer

        if hasattr(rag_response, "citations"):
            citations = rag_response.citations

    # Format and return the answer
    if not answer:
        return "No answer was generated."

    # Format the output
    lines = []
    lines.append(f"{answer}")

    # Format citations if available
    if citations and len(citations) > 0:
        lines.append("\nCitations:")
        for i, citation in enumerate(citations, 1):
            if isinstance(citation, dict):
                text = citation.get("text", "").strip()
                source = citation.get("source", "Unknown source")
                lines.append(f"\n- {text} (Source: {source})")
            elif hasattr(citation, "payload"):
                citation_id = getattr(citation, "id", f"citation_{i}")

                # Extract citation text based on payload type
                if isinstance(citation.payload, str):
                    text = citation.payload
                elif (
                    isinstance(citation.payload, dict)
                    and "text" in citation.payload
                ):
                    text = citation.payload["text"]
                else:
                    text = str(citation.payload)

                lines.append(f"\n[{citation_id}] {text}")
            elif isinstance(citation, str):
                lines.append(f"\n- {citation}")
            else:
                # Handle any other type of citation
                text = str(citation)
                lines.append(f"\n- {text}")

    return "\n".join(lines)


class R2RMCPServer:
    """R2R MCP Server implementation."""

    def __init__(self):
        """Initialize the R2R MCP Server."""
        self.mcp = FastMCP("R2R Retrieval System")
        # Add tools property to FastMCP for testing compatibility
        self.mcp._tools = []
        self._tools = []
        self.logger = logger
        self.client = None
        self.async_client = None

        # Log configuration
        api_url = os.environ.get("R2R_API_URL")
        api_key = os.environ.get("R2R_API_KEY")

        if api_url:
            logger.info(f"Using R2R API URL: {api_url}")
        elif api_key:
            logger.info("Using R2R API with provided API key")
        else:
            logger.warning(
                "No R2R_API_URL or R2R_API_KEY provided. Using default configuration."
            )

        # Set up tools after initialization
        self.setup_tools()

    @property
    def tools(self):
        """Get the list of tools registered with the MCP server.

        This ensures compatibility with the test suite that expects the tools
        to be accessible via the server.tools attribute.
        """
        return self._tools

    def setup_tools(self):
        """Set up MCP tools."""
        # Reset tools list
        self._tools = []
        self.mcp._tools = []

        # Define and register tools
        @self.mcp.tool()
        async def search(query: str) -> str:
            """Performs a search across vector, graph, web, and document sources."""
            logger.info(f"Performing search: {query}")

            try:
                client = self.get_client()

                # Configure search settings
                search_settings = {
                    "limit": 10,
                    "use_hybrid_search": True,
                    "full_text_search": True,
                    "full_text_search_config": {"limit": 200},
                }

                # Call the search endpoint
                search_response = client.retrieval.search(
                    query=query, search_settings=search_settings
                )

                # Format and return results
                formatted_results = format_search_results_for_llm(
                    search_response.results
                )
                logger.info(f"Search completed successfully for: {query}")
                return formatted_results

            except Exception as e:
                error_msg = f"Error performing search: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"Search failed: {error_msg}"

        search.name = "search"
        search.func = search
        self._tools.append(search)
        self.mcp._tools.append(search)

        @self.mcp.tool()
        async def rag(query: str) -> str:
            """Perform a Retrieval-Augmented Generation query."""
            logger.info(f"Performing RAG query: {query}")

            try:
                client = self.get_client()

                # Configure search settings
                search_settings = {
                    "limit": 10,
                    "use_hybrid_search": True,
                    "full_text_search": True,
                    "full_text_search_config": {"limit": 200},
                }

                # Configure RAG generation
                rag_generation_config = {
                    "temperature": 0.7,
                    "max_tokens_to_sample": 1000,
                }

                # Call the RAG endpoint
                rag_response = client.retrieval.rag(
                    query=query,
                    search_settings=search_settings,
                    rag_generation_config=rag_generation_config,
                )

                # Format and return the response
                formatted_response = format_rag_response(rag_response)
                logger.info(f"RAG completed successfully for: {query}")
                return formatted_response

            except Exception as e:
                error_msg = f"Error performing RAG: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"RAG failed: {error_msg}"

        rag.name = "rag"
        rag.func = rag
        self._tools.append(rag)
        self.mcp._tools.append(rag)

        @self.mcp.tool()
        async def web_search(query: str) -> str:
            """Perform a web search to find information online."""
            logger.info(f"Performing web search: {query}")

            try:
                client = self.get_client()

                # Use the search endpoint with web search settings
                # Apparently, the client doesn't have a web_search method directly
                search_settings = {
                    "web_search": True,
                    "limit": 10,
                }  # Enable web search

                # Call the search endpoint with web search configuration
                web_response = client.retrieval.search(
                    query=query, search_settings=search_settings
                )

                # Format and return results
                if hasattr(web_response, "results"):
                    results = web_response.results
                else:
                    results = web_response

                # Extract web results
                web_results = []
                if hasattr(results, "web_search_results"):
                    web_results = results.web_search_results

                # Format results
                lines = ["## Web Search Results:"]

                if web_results:
                    for i, result in enumerate(web_results[:10], 1):
                        title = getattr(result, "title", "Untitled")
                        link = getattr(result, "link", "No link")
                        snippet = getattr(result, "snippet", "No snippet")

                        lines.append(f"{i}. **{title}**")
                        lines.append(f"   Link: {link}")
                        lines.append(f"   {snippet}")
                        lines.append("")
                else:
                    lines.append("No web search results found.")

                logger.info(f"Web search completed successfully for: {query}")
                return "\n".join(lines)

            except Exception as e:
                error_msg = f"Error performing web search: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"Web search failed: {error_msg}"

        web_search.name = "web_search"
        web_search.func = web_search
        self._tools.append(web_search)
        self.mcp._tools.append(web_search)

        @self.mcp.tool()
        async def document_search(
            query: str, document_id: Optional[str] = None
        ) -> str:
            """Search within specific documents or across all documents."""
            logger.info(f"Performing document search: {query}")
            if document_id:
                logger.info(f"Restricting to document ID: {document_id}")

            try:
                client = self.get_client()

                # Configure search settings
                search_settings = {
                    "limit": 10,
                    "use_hybrid_search": True,
                    "full_text_search": True,
                    "full_text_search_config": {"limit": 200},
                }

                # Add document filter if specified
                if document_id:
                    search_settings["filters"] = {
                        "document_id": {"$eq": document_id}
                    }

                # Call the search endpoint
                search_response = client.retrieval.search(
                    query=query, search_settings=search_settings
                )

                # Format and return results
                formatted_results = format_search_results_for_llm(
                    search_response.results
                )
                logger.info(
                    f"Document search completed successfully for: {query}"
                )
                return formatted_results

            except Exception as e:
                error_msg = f"Error performing document search: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"Document search failed: {error_msg}"

        document_search.name = "document_search"
        document_search.func = document_search
        self._tools.append(document_search)
        self.mcp._tools.append(document_search)

        @self.mcp.tool()
        async def list_documents() -> str:
            """List available documents in the knowledge base."""
            logger.info("Listing documents")

            try:
                client = self.get_client()

                # Call the list documents endpoint
                documents_result = client.documents.list(limit=20)

                # Process documents result
                documents = []
                if hasattr(documents_result, "results") and hasattr(
                    documents_result.results, "items"
                ):
                    documents = documents_result.results.items
                elif hasattr(documents_result, "results"):
                    documents = documents_result.results
                elif hasattr(documents_result, "data"):
                    documents = documents_result.data
                elif isinstance(documents_result, list):
                    documents = documents_result

                # Format results
                lines = ["## Available Documents:"]

                if documents:
                    for i, doc in enumerate(documents, 1):
                        doc_id = getattr(doc, "id", "Unknown")
                        doc_title = "Untitled"

                        if hasattr(doc, "title"):
                            doc_title = doc.title
                        elif hasattr(doc, "metadata") and doc.metadata:
                            if isinstance(doc.metadata, dict):
                                doc_title = doc.metadata.get(
                                    "title", "Untitled"
                                )

                        doc_type = getattr(doc, "document_type", "Unknown")
                        doc_status = getattr(doc, "status", "Unknown")

                        lines.append(f"{i}. **{doc_title}**")
                        lines.append(f"   ID: {doc_id}")
                        lines.append(f"   Type: {doc_type}")
                        lines.append(f"   Status: {doc_status}")
                        lines.append("")
                else:
                    lines.append("No documents found in the knowledge base.")

                logger.info(f"Found {len(documents)} documents")
                return "\n".join(lines)

            except Exception as e:
                error_msg = f"Error listing documents: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"Document listing failed: {error_msg}"

        list_documents.name = "list_documents"
        list_documents.func = list_documents
        self._tools.append(list_documents)
        self.mcp._tools.append(list_documents)

        @self.mcp.tool()
        async def agent_research(query: str) -> str:
            """Perform deep research on a topic using the agent mode."""
            logger.info(f"Performing agent research: {query}")

            try:
                client = self.get_client()

                # Configure search settings
                search_settings = {
                    "limit": 10,
                    "use_hybrid_search": True,
                    "full_text_search": True,
                }

                # Configure agent generation
                research_generation_config = {
                    "model": "gpt-4o",  # Use a powerful model for research
                    "temperature": 0.7,
                    "max_tokens_to_sample": 2000,
                    "stream": False,
                }

                # Call the agent endpoint in research mode
                agent_response = client.retrieval.agent(
                    message={"role": "user", "content": query},
                    search_settings=search_settings,
                    research_generation_config=research_generation_config,
                    research_tools=["rag", "reasoning", "critique"],
                    mode="research",
                )

                # Extract the response
                if hasattr(agent_response, "results") and hasattr(
                    agent_response.results, "messages"
                ):
                    final_message = agent_response.results.messages[-1].content
                    logger.info(
                        f"Agent research completed successfully for: {query}"
                    )
                    return final_message
                else:
                    logger.warning(
                        f"Unexpected agent response format: {agent_response}"
                    )
                    return "Research completed, but the response format was unexpected."

            except Exception as e:
                error_msg = f"Error performing agent research: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"Agent research failed: {error_msg}"

        agent_research.name = "agent_research"
        agent_research.func = agent_research
        self._tools.append(agent_research)
        self.mcp._tools.append(agent_research)

    def get_client(self):
        """Get or create an R2R client."""
        if self.client is None:
            api_url = os.environ.get("R2R_API_URL")
            api_key = os.environ.get("R2R_API_KEY")

            try:
                if api_url:
                    self.client = R2RClient(base_url=api_url)
                elif api_key:
                    self.client = R2RClient()
                    self.client.api_key = api_key
                else:
                    self.client = R2RClient()

                logger.info("R2R client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize R2R client: {e}")
                raise

        return self.client

    async def get_async_client(self):
        """Get or create an async R2R client."""
        if self.async_client is None:
            api_url = os.environ.get("R2R_API_URL")
            api_key = os.environ.get("R2R_API_KEY")

            try:
                if api_url:
                    self.async_client = R2RAsyncClient(base_url=api_url)
                elif api_key:
                    self.async_client = R2RAsyncClient()
                    self.async_client.api_key = api_key
                else:
                    self.async_client = R2RAsyncClient()

                logger.info("R2R async client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize R2R async client: {e}")
                raise

        return self.async_client

    def register_tools(self):
        """Register tools with the MCP server. Used for backwards compatibility."""
        # Already registered in setup_tools
        pass

    def run(self):
        """Run the MCP server."""
        logger.info("Starting R2R MCP Server")
        self.mcp.run()


# Run the server if executed directly
if __name__ == "__main__":
    server = R2RMCPServer()
    server.run()
