# rizzbot_agentic.py with logging

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langsmith import Client
from langchain.retrievers.multi_query import MultiQueryRetriever


class Rizzbot:
    def __init__(self):
        print("[INIT] Starting Rizzbot initialization...")
        _ = self._load_env()
        self.similarity_threshold = 0.3
        self.top_k = 3
        self.summary_threshold = 3  # Stop after finding this many docs in summaries

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "rizzbot"
        print("[ENV] Environment variables set.")

        self.client = Client()

        self.main_llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
        print("[LLM] Main LLM (gpt-4o) initialized.")

        self.expand_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        print("[LLM] Expansion LLM (gpt-3.5-turbo) initialized.")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("[Embeddings] OpenAI embeddings initialized.")

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("[Pinecone] Pinecone client initialized.")

        self.summaries_vectorstore = PineconeVectorStore(
            index_name="rizzbot-summaries-full-text",
            embedding=self.embeddings,
            text_key="full_text"
        )
        print("[VectorStore] Summaries vector store initialized.")

        self.full_vectorstore = PineconeVectorStore(
            index_name="rizzbot", embedding=self.embeddings, text_key="full_text"
        )
        print("[VectorStore] Full vector store initialized.")

        self.no_answer_response = "Sorry bro, I couldn't find enough info to answer that confidently."

        self.base_prompt_template = ChatPromptTemplate.from_template("""
        You are a charisma and personal development expert helping someone improve their social skills.

        Context: {content}
        Question: {question}

        Instructions:
        1. Analyze the question and context. Check the vectorstores for an answer. If the answer can not be found in the vectorstore, answer: "Sorry bro, I couldn't find enough info in my database to answer that confidently.""
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
            # Try to get source information from metadata
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

        # First, try summaries vectorstore
        print(f"[Search:Hybrid] Trying summaries vectorstore...")
        try:
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.summaries_vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
                llm=self.expand_llm
            )
            docs = retriever.invoke(question)
            filtered, sources = self._filter_by_similarity(question_embedding, docs, self.similarity_threshold)
            print(f"[Search:Hybrid] {len(filtered)} docs passed threshold in summaries.")
            
            if len(filtered) > self.summary_threshold:
                print(f"[Search:Hybrid] Found {len(filtered)} docs in summaries (>{self.summary_threshold}), skipping full search.")
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
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.full_vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
                llm=self.expand_llm
            )
            docs = retriever.invoke(question)
            filtered, sources = self._filter_by_similarity(question_embedding, docs, self.similarity_threshold)
            print(f"[Search:Hybrid] {len(filtered)} docs passed threshold in full.")
            combined_results.extend([doc.page_content for doc in filtered])
            all_sources.extend(sources)
        except Exception as e:
            print(f"[Search:Hybrid] Retrieval failed for full: {e}")

        return combined_results, all_sources

    def _build_agent_chain(self):
        print("[Chain] Building agent chain...")

        def hybrid_search_with_sources(q):
            results, sources = self._hybrid_query_search(q)
            if results:
                content = "\n\n".join(results)
                # Add sources to the content for the LLM to use
                if sources:
                    content += f"\n\nSources: {', '.join(set(sources))}"
                return content
            else:
                return self.no_answer_response
    
        self.agent_chain = (
            {
                "question": lambda q: q,
                "content": hybrid_search_with_sources,
            }
            | self.base_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        print("[Chain] Agent chain constructed.")

    def answer_question(self, question: str) -> str:
        print(f"[Answer] Received question: {question}")
        try:
            context, sources = self._hybrid_query_search(question)
            if not context:
                print("[Answer] No relevant documents found. Returning fallback response.")
                return self.no_answer_response

            print("[Answer] Relevant context found. Generating response with LLM...")
            answer = self.agent_chain.invoke(question)
            print(f"[Answer] Answer generated successfully.")
            return answer
        except Exception as e:
            print(f"[Answer] Agentic pipeline failed: {e}")
            return self.no_answer_response

