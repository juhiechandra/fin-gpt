import requests
import json
from typing import Optional, List, Dict, Any
from logger import model_logger, error_logger
import time


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = "jjansen/adapt-finance-llama2-7b:latest"

    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            model_logger.error(f"Ollama not available: {e}")
            return False

    def chat_completion(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request to Ollama"""
        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context, chat_history)

            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 2048,
                },
            }

            model_logger.info(
                f"Sending request to Ollama with model: {self.model_name}"
            )
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,  # 2 minute timeout
            )

            end_time = time.time()
            processing_time = end_time - start_time

            if response.status_code == 200:
                result = response.json()
                model_logger.info(f"Ollama response received in {processing_time:.2f}s")
                return {
                    "response": result.get("response", ""),
                    "processing_time": processing_time,
                    "model": self.model_name,
                    "success": True,
                }
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                model_logger.error(error_msg)
                return {
                    "response": "Sorry, there was an error processing your request with the finance model.",
                    "processing_time": processing_time,
                    "model": self.model_name,
                    "success": False,
                    "error": error_msg,
                }

        except Exception as e:
            error_msg = f"Error calling Ollama: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return {
                "response": "Sorry, the finance model is currently unavailable.",
                "processing_time": 0,
                "model": self.model_name,
                "success": False,
                "error": error_msg,
            }

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List] = None,
    ) -> str:
        """Build the full prompt for the model"""

        # Start with system prompt for finance domain
        system_prompt = """You are a helpful financial assistant with expertise in finance, investments, and business analysis. You provide accurate, professional, and easy-to-understand financial advice and information."""

        full_prompt = system_prompt + "\n\n"

        # Add chat history if provided
        if chat_history and len(chat_history) > 0:
            full_prompt += "Previous conversation:\n"
            for msg in chat_history[-6:]:  # Keep last 6 messages for context
                if isinstance(msg, dict):
                    if msg.get("role") == "human" or msg.get("type") == "human":
                        full_prompt += (
                            f"Human: {msg.get('content', msg.get('question', ''))}\n"
                        )
                    elif msg.get("role") == "assistant" or msg.get("type") == "ai":
                        full_prompt += (
                            f"Assistant: {msg.get('content', msg.get('answer', ''))}\n"
                        )
                elif isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    if role == "human":
                        full_prompt += f"Human: {content}\n"
                    elif role == "ai":
                        full_prompt += f"Assistant: {content}\n"
            full_prompt += "\n"

        # Add context from RAG if provided
        if context and context.strip():
            full_prompt += f"Relevant information from documents:\n{context}\n\n"

        # Add the current question
        full_prompt += f"Human: {prompt}\nAssistant: "

        return full_prompt


def get_ollama_rag_chain(use_hybrid_search: bool = True):
    """Create a RAG chain using Ollama finance model"""
    from faiss_utils import vectorstore
    from hybrid_search import create_hybrid_retriever_from_faiss

    model_logger.info("Creating Ollama RAG chain with finance model")

    # Initialize Ollama client
    ollama_client = OllamaClient()

    # Check if Ollama is available
    if not ollama_client.is_available():
        raise Exception("Ollama server is not available. Please start Ollama first.")

    # Configure retriever
    if use_hybrid_search:
        model_logger.info("Using hybrid search (vector + BM25) for Ollama")
        retriever = create_hybrid_retriever_from_faiss(
            vectorstore=vectorstore,
            k=6,
            weight_vector=0.6,
            weight_keyword=0.4,
            use_rrf=True,
            rrf_k=60,
        )
    else:
        model_logger.info("Using vector search only for Ollama")
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.75},
        )

    class OllamaRAGChain:
        """Custom RAG chain for Ollama"""

        def __init__(self, ollama_client, retriever):
            self.ollama_client = ollama_client
            self.retriever = retriever

        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Process the RAG query"""
            question = inputs.get("input", "")
            chat_history = inputs.get("chat_history", [])

            # Retrieve relevant documents
            try:
                docs = self.retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in docs])
                model_logger.info(f"Retrieved {len(docs)} documents for context")
            except Exception as e:
                model_logger.warning(f"Error retrieving documents: {e}")
                context = ""

            # Get response from Ollama
            result = self.ollama_client.chat_completion(
                prompt=question, context=context, chat_history=chat_history
            )

            return {
                "answer": result["response"],
                "processing_time": result["processing_time"],
                "model": result["model"],
            }

    return OllamaRAGChain(ollama_client, retriever)


def get_ollama_simple_chat():
    """Create a simple chat interface with Ollama (no RAG)"""
    model_logger.info("Creating simple Ollama chat (no RAG)")

    # Initialize Ollama client
    ollama_client = OllamaClient()

    # Check if Ollama is available
    if not ollama_client.is_available():
        raise Exception("Ollama server is not available. Please start Ollama first.")

    class OllamaSimpleChat:
        """Simple chat interface for Ollama"""

        def __init__(self, ollama_client):
            self.ollama_client = ollama_client

        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Process the simple chat query"""
            question = inputs.get("input", "")
            chat_history = inputs.get("chat_history", [])

            # Get response from Ollama without RAG
            result = self.ollama_client.chat_completion(
                prompt=question,
                context=None,  # No RAG context
                chat_history=chat_history,
            )

            return {
                "answer": result["response"],
                "processing_time": result["processing_time"],
                "model": result["model"],
            }

    return OllamaSimpleChat(ollama_client)
