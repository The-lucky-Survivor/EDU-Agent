"""
Phase 5: LLM Integration & Response Generation
Handles LLM setup, RAG chain assembly, and response post-processing.
Supports Groq's openai/gpt-oss-120b reasoning model.
"""

import os
import re
import logging
from typing import Dict, Optional, Union

from src.retrieval import (
    EDUCATIONAL_PROMPT,
    EDUCATIONAL_PROMPT_AR,
    get_basic_retriever,
    get_advanced_retriever,
    format_docs,
    format_docs_ar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def get_llm(provider: str = None, model: str = None, temperature: float = None):
    """
    Initialize LLM based on configuration.
    Falls back through providers if API keys are not available.
    
    Priority: Groq -> OpenAI -> Fake (for testing)
    """
    from langchain_core.language_models import BaseChatModel
    
    # Try Groq first (primary provider)
    groq_key = os.getenv('GROQ_API_KEY', '')
    if groq_key and groq_key != 'your_groq_api_key_here':
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=model or 'openai/gpt-oss-120b',
                temperature=temperature if temperature is not None else 1.0,
                groq_api_key=groq_key,
                max_tokens=int(os.getenv('GROQ_MAX_COMPLETION_TOKENS', '4096')),
                reasoning_effort=os.getenv('GROQ_REASONING_EFFORT', 'medium'),
            )
            logger.info(f"✅ Groq LLM initialized: {model or 'openai/gpt-oss-120b'}")
            return llm
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")
    
    # Try OpenAI as fallback
    openai_key = os.getenv('OPENAI_API_KEY', '')
    if openai_key and openai_key != 'your_openai_api_key_here':
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model or 'gpt-4o-mini',
                temperature=temperature if temperature is not None else 0.1,
                openai_api_key=openai_key,
            )
            logger.info(f"✅ OpenAI LLM initialized: {llm.model_name}")
            return llm
        except Exception as e:
            logger.warning(f"OpenAI init failed: {e}")
    
    # Fallback: Use a simple mock for testing
    logger.warning("⚠️ No LLM API key found. Using mock LLM (responses will be simulated).")
    return MockLLM()


class MockLLM:
    """
    Mock LLM for testing without API keys.
    Returns structured responses based on the context provided.
    """
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def invoke(self, messages):
        """Generate a mock response."""
        # Handle different message formats
        content = ""
        if isinstance(messages, list) and len(messages) > 0:
            # List of messages - extract content
            content = "\n".join([m.content if hasattr(m, 'content') else str(m) for m in messages])
        elif hasattr(messages, 'content'):
            content = messages.content
        elif hasattr(messages, 'to_string'):
            content = messages.to_string()
        else:
            content = str(messages)
        
        # Extract question (after "Student Question:" or "سؤال الطالب:")
        question = "your question"
        if "Student Question:" in content:
            q_part = content.split("Student Question:")[-1]
            # Stop at "Your Answer:" if present
            if "Your Answer:" in q_part:
                q_part = q_part.split("Your Answer:")[0]
            question = q_part.strip().split("\n")[0].strip()
        elif "سؤال الطالب:" in content:
            q_part = content.split("سؤال الطالب:")[-1]
            if "إجابتك:" in q_part:
                q_part = q_part.split("إجابتك:")[0]
            question = q_part.strip().split("\n")[0].strip()
        
        # Extract context snippets
        context_snippets = []
        if "Source:" in content:
            sources = re.findall(r'\[Source: ([^\]]+)\]', content)
            context_snippets = list(set(sources))
        elif "المصدر:" in content:
            sources = re.findall(r'\[المصدر: ([^\]]+)\]', content)
            context_snippets = list(set(sources))
        
        # Build response
        response = f"""Based on the lecture materials provided, here's what I found about \"{question}\":

• The lectures cover this topic in detail across multiple chapters.

Key Points from the Lectures:
• The content discusses agent architectures, reinforcement learning, and decision-making processes.
• Specific examples and algorithms are explained with step-by-step breakdowns.

"""
        
        if context_snippets:
            response += "\nSources Referenced:\n"
            for src in context_snippets[:3]:
                response += f"• {src}\n"
        
        response += "\nNote: I'm currently running in demo mode without a real LLM API key. To get AI-generated answers, please set your GROQ_API_KEY in the .env file."
        
        from langchain_core.messages import AIMessage
        return AIMessage(content=response)
    
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


# =============================================================================
# RAG CHAIN
# =============================================================================

def create_rag_chain(vectorstore, llm=None, use_arabic: bool = False, use_advanced: bool = False):
    """
    Create a complete RAG chain.
    
    Args:
        vectorstore: Chroma vector store
        llm: LLM instance (creates default if None)
        use_arabic: Use Arabic prompt template
        use_advanced: Use advanced retriever with re-ranking
    
    Returns:
        Configured RAG chain (callable)
    """
    if llm is None:
        llm = get_llm()
    
    # Select retriever
    if use_advanced:
        retriever = get_advanced_retriever(vectorstore)
    else:
        retriever = get_basic_retriever(vectorstore)
    
    # Select prompt
    prompt = EDUCATIONAL_PROMPT_AR if use_arabic else EDUCATIONAL_PROMPT
    format_fn = format_docs_ar if use_arabic else format_docs
    
    # Build chain manually (compatible with all LangChain versions)
    def rag_chain(question: str) -> str:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        
        # Format context
        context = format_fn(docs)
        
        # Format prompt
        messages = prompt.format_messages(context=context, question=question)
        
        # Generate response
        response = llm.invoke(messages)
        
        # Extract content
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    logger.info("✅ RAG chain created successfully")
    return rag_chain


# =============================================================================
# RESPONSE POST-PROCESSING
# =============================================================================

def post_process_answer(answer: str) -> Dict:
    """
    Post-process the LLM answer to extract sources and confidence.
    
    Args:
        answer: Raw LLM response
    
    Returns:
        Processed answer with metadata
    """
    sources = []
    lines = answer.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if any(marker in line_lower for marker in ['source:', 'المصدر:', 'صفحة', 'page']):
            sources.append(line.strip())
    
    # Determine confidence
    has_sources = len(sources) > 0
    is_refusal = any(phrase in answer.lower() for phrase in [
        'not available', 'غير موجود', 'not found', 'not in the lectures'
    ])
    
    confidence = 'high' if has_sources and not is_refusal else \
                 'medium' if not is_refusal else 'low'
    
    return {
        'answer': answer.strip(),
        'sources': sources,
        'confidence': confidence,
        'is_refusal': is_refusal,
    }
