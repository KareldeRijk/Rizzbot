# rizzbot_agentic.py with logging - version used for final project, changed to use 384 vector dimensions and only one openAI model instead of three

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langsmith import Client
'''from langchain_community.retrievers.multi_query import MultiQueryRetriever'''
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

class Rizzbot:
    def __init__(self):
        print("[INIT] Starting Rizzbot initialization...")
        _ = self._load_env()
        self.similarity_threshold = 0.3
        self.top_k = 3
        self.summary_threshold = 1  # Stop after finding this many docs in summaries
        self.min_docs_threshold = 1  # Minimum docs required to attempt answer generation

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "rizzbot"
        print("[ENV] Environment variables set.")

        self.client = Client()

        # Use a proper text generation model instead of sentence transformer for text generation
        text_generation_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",  # Using a proper text generation model
            tokenizer="microsoft/DialoGPT-small",
            max_length=500,
            temperature=0.6,
            do_sample=True,
            device=0 if torch.cuda.is_available() else -1,
            pad_token_id=50256  # Add pad token for proper tokenization
        )
        self.main_llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
        print("[LLM] Main LLM (gpt-4o) initialized.")
        
        self.expand_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)    
        print("[LLM] Expansion LLM (all-MiniLM-L6-v2) initialized.")

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("[Embeddings] SentenceTransformer embeddings model initialized.")

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("[Pinecone] Pinecone client initialized.")

        self.summaries_vectorstore = PineconeVectorStore(
            index=self.pc.Index("rizzbot-summaries-full-text-384"),
            embedding=self.embeddings,
            text_key="text"
        )
        print("[VectorStore] Summaries vector store initialized.")

        self.full_vectorstore = PineconeVectorStore(
            index=self.pc.Index("rizzbot-384"), embedding=self.embeddings, text_key="text"
        )
        print("[VectorStore] Full vector store initialized.")

        self.no_answer_response = "Apologies, I couldn't find enough info in my database to answer that confidently."

        self.base_prompt_template = ChatPromptTemplate.from_template("""
        You are a charisma and personal development expert helping someone improve their social skills.

        Context: {content}
        Question: {question}

        Instructions:
        1. Analyze the question and context. Base your answer on the information available in the vectorstore texts only. 
        2. If the question is not clear, ask for clarification.
        3. If the question is clear, provide actionable, specific advice based on the context.
        4. Use examples when possible
        5. Keep the tone encouraging and supportive
        6. If information is insufficient, explain what you'd need to give a better answer
        7. At the end of your response, include a "Sources:" section listing the document sources used

        Response:
        """)

        self._build_agent_chain()
        print("[INIT] Rizzbot initialized and ready.")

    def _load_env(self):
        from dotenv import load_dotenv, find_dotenv
        print("[ENV] Loading environment variables from .env file...")
        return load_dotenv(find_dotenv())

    def _embed_question(self, question: str) -> List[float]:
        print(f"[Embed] Embedding question: {question}")
        result = self.embeddings.embed_query(question)
        print(f"[Embed] Embedding result length: {len(result)}")
        return result

    def _cosine_similarity(self, vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _filter_by_similarity(self, query_embedding, docs, threshold):
        filtered = []
        sources = []

        for doc in docs:
            try:
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                sim = self._cosine_similarity(query_embedding, doc_embedding)
                print(f"[Similarity] Score: {sim:.4f} | Text: {doc.page_content[:80]}...")

                if sim >= threshold:
                    filtered.append(doc)
                    # Extract source information from document metadata
                    source_info = self._extract_source_info(doc)
                    sources.append(source_info)
            except Exception as e:
                print(f"[Similarity] Failed to embed doc: {e}")

        return filtered, sources

    def _extract_source_info(self, doc) -> str:
        """Extract source information from document metadata"""
        if hasattr(doc, 'metadata') and doc.metadata:
            # Priority 1: Try to get topic_id
            topic_id = doc.metadata.get('topic_id')
            if topic_id is not None:
                return f"Topic {topic_id}"
            
            # Priority 2: Try to get document ID
            doc_id = doc.metadata.get('id') or doc.metadata.get('doc_id') or doc.metadata.get('document_id')
            if doc_id is not None:
                return f"Document {doc_id}"
            
            # Priority 3: Try to get source information
            source = doc.metadata.get('source', 'Unknown source')
            title = doc.metadata.get('title', '')
            if title:
                return f"{title} ({source})"
            else:
                return source
        else:
            # Fallback to truncated content as identifier
            return f"Document: {doc.page_content[:50]}..."

    def _hybrid_query_search(self, question: str) -> Tuple[List[str], List[str]]:
        print(f"[Search:Hybrid] Embedding question...")
        question_embedding = self._embed_question(question)
        combined_results = []
        all_sources = []
    
        # First, try summaries vectorstore with direct search (MultiQueryRetriever removed)
        print(f"[Search:Hybrid] Trying summaries vectorstore...")
        try:
            docs = self.summaries_vectorstore.similarity_search(question, k=self.top_k)
            print(f"[Search:Hybrid] Direct search succeeded for summaries")
                
            filtered, sources = self._filter_by_similarity(question_embedding, docs, self.similarity_threshold)
            print(f"[Search:Hybrid] {len(filtered)} docs passed threshold in summaries.")
            
            if len(filtered) >= self.summary_threshold:
                print(f"[Search:Hybrid] Found {len(filtered)} docs in summaries (>={self.summary_threshold}), skipping full search.")
                combined_results.extend([doc.page_content for doc in filtered])
                all_sources.extend(sources)
                return combined_results, all_sources
            else:
                combined_results.extend([doc.page_content for doc in filtered])
                all_sources.extend(sources)
        except Exception as e:
            print(f"[Search:Hybrid] Retrieval failed for summaries: {e}")
    
        # If we didn't find enough in summaries, search full vectorstore
        print(f"[Search:Hybrid] Trying full vectorstore...")
        try:
            docs = self.full_vectorstore.similarity_search(question, k=self.top_k)
            print(f"[Search:Hybrid] Direct search succeeded for full")
                
            filtered, sources = self._filter_by_similarity(question_embedding, docs, self.similarity_threshold)
            print(f"[Search:Hybrid] {len(filtered)} docs passed threshold in full.")
            combined_results.extend([doc.page_content for doc in filtered])
            all_sources.extend(sources)
        except Exception as e:
            print(f"[Search:Hybrid] Retrieval failed for full: {e}")
    
        return combined_results, all_sources
    
    def _build_agent_chain(self):
        print("[Chain] Building agent chain...")

        def format_content_with_sources(content_and_sources):
            """Format content with sources for the LLM"""
            content, sources = content_and_sources
            if content:
                formatted_content = content
                if sources:
                    formatted_content += f"\n\nSources: {', '.join(set(sources))}"
                return formatted_content
            else:
                return self.no_answer_response
    
        self.agent_chain = (
            {
                "question": lambda q: q,
                "content": format_content_with_sources,
            }
            | self.base_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        print("[Chain] Agent chain constructed.")

    def answer_question(self, question: str) -> str:
        print(f"[Answer] Received question: {question}")
        try:
            # Perform search once
            context, sources = self._hybrid_query_search(question)
            
            # Check if we have enough documents
            if not context or len(context) < self.min_docs_threshold:
                print(f"[Answer] Insufficient documents found ({len(context) if context else 0} docs, need {self.min_docs_threshold}). Returning fallback response.")
                return self.no_answer_response

            print(f"[Answer] Found {len(context)} relevant documents. Generating response with LLM...")
            
            # Format content with sources
            content_text = "\n\n".join(context)
            if sources:
                content_text += f"\n\nSources: {', '.join(set(sources))}"
            
            # Pass the pre-searched content to the chain
            answer = self.agent_chain.invoke({
                "question": question,
                "content": (content_text, sources)
            })
            
            print(f"[Answer] Answer generated successfully.")
            return answer
        except Exception as e:
            print(f"[Answer] Agentic pipeline failed: {e}")
            return self.no_answer_response
