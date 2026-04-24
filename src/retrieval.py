"""
Phase 4: RAG Pipeline Construction
Handles retrieval, re-ranking, and prompt engineering.
"""

import os
import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.config import (
    RETRIEVER_TOP_K,
    RETRIEVER_FINAL_K,
    USE_RERANKING,
    COHERE_API_KEY,
)

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

EDUCATIONAL_PROMPT = ChatPromptTemplate.from_template('''
You are an educational assistant specialized in Artificial Intelligence and Multi-Agent Systems.
Your task is to answer student questions based ONLY on the provided lecture content.

Important Rules:
1. If the answer is in the lectures, explain it clearly and accurately.
2. If the answer is NOT in the lectures, clearly state: "This information is not available in the provided lectures."
3. Do NOT make up information. Do NOT use your general knowledge unless explicitly asked.
4. Cite the source (file name and page number) at the end of your answer.
5. If the question requires examples, use examples from the lectures.
6. Make your answer organized using bullet points or numbered lists when appropriate.

Lecture Content:
{context}

Student Question:
{question}

Your Answer:
''')

# Arabic prompt version (if needed)
EDUCATIONAL_PROMPT_AR = ChatPromptTemplate.from_template('''
أنت مساعد تعليمي متخصص في الذكاء الاصطناعي وأنظمة الوكلاء المتعددين.
مهمتك هي الإجابة على أسئلة الطلاب بناءً على المحاضرات المقدمة فقط.

قواعد مهمة:
1. إذا كانت الإجابة في المحاضرات، اشرحها بوضوح ودقة.
2. إذا لم تكن الإجابة في المحاضرات، قول بوضوح: 'هذه المعلومة غير موجودة في المحاضرات المتاحة.'
3. لا تخترع معلومات. لا تستخدم معرفتك العامة إلا إذا طُلب منك ذلك صراحةً.
4. اذكر المصدر (اسم الملف ورقم الصفحة) في نهاية إجابتك.
5. إذا كان السؤال يتطلب أمثلة، استخدم الأمثلة من المحاضرات.
6. اجعل الإجابة منظمة باستخدام النقاط أو الأرقام عند الحاجة.

المحتوى من المحاضرات:
{context}

سؤال الطالب:
{question}

إجابتك:
''')


# =============================================================================
# RETRIEVER SETUP
# =============================================================================

def get_basic_retriever(vectorstore, k: int = 8):
    """Get basic similarity-based retriever."""
    return vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )


def get_advanced_retriever(vectorstore, top_k: int = 10, final_k: int = 5):
    """Get advanced retriever with optional Cohere re-ranking."""
    base_retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': top_k}
    )
    
    if USE_RERANKING and COHERE_API_KEY:
        try:
            from langchain_cohere import CohereRerank
            from langchain.retrievers import ContextualCompressionRetriever
            
            compressor = CohereRerank(
                cohere_api_key=COHERE_API_KEY,
                top_n=final_k
            )
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            logger.info("Advanced retriever with Cohere re-ranking initialized")
            return compression_retriever
        except Exception as e:
            logger.warning(f"Failed to initialize re-ranking: {e}. Using basic retriever.")
    
    return base_retriever


# =============================================================================
# DOCUMENT FORMATTING
# =============================================================================

def format_docs(docs: List[Document]) -> str:
    """Format documents for prompt context with source citations."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page_number', 'N/A')
        formatted.append(f'[Source: {source} - Page {page}]\n{doc.page_content}')
    return '\n\n---\n\n'.join(formatted)


def format_docs_ar(docs: List[Document]) -> str:
    """Format documents with Arabic citations."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page_number', 'N/A')
        formatted.append(f'[المصدر: {source} - صفحة {page}]\n{doc.page_content}')
    return '\n\n---\n\n'.join(formatted)
