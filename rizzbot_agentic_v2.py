# rizzbot_agentic_improved.py

import os
import logging
from typing import List, Dict, Optional
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langsmith import Client
from langchain.retrievers.multi_query import MultiQueryRetriever
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Rizzbot:
    def __init__(self):
        logger.info(" Initializing Rizzbot...")
        _ = self._load_env()
        self.similarity_threshold = 0.85
        self.top_k = 3

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "rizzbot"
        self.client = Client()

        logger.info(" Setting up LLM models...")
        self.main_llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
        self.expand_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        logger.info(" Connecting to Pinecone vector stores...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.summaries_vectorstore = PineconeVectorStore(
            index_name="rizzbot-summaries-full-text",
            embedding=self.embeddings,
            text_key="full_text"
        )
        self.full_vectorstore = PineconeVectorStore(
            index_name="rizzbot", embedding=self.embeddings, text_key="full_text"
        )

        logger.info(" Setting up prompt templates...")
        self.base_prompt_template = ChatPromptTemplate.from_template("""
        You are a charisma and personal development expert.

        Based on the following content, answer the user's question clearly and helpfully.
        If the content doesn't contain enough information to answer the question confidently, 
        say so and ask for clarification or suggest related topics you can help with.

        User Question: {question}

        Relevant Content:
        {content}

        Confidence Score: {confidence}

        Answer:
        """)

        # Enhanced prompt for better responses
        self.enhanced_prompt_template = ChatPromptTemplate.from_template("""
        You are a charisma and personal development expert helping someone improve their social skills.

        Context: {content}
        Question: {question}
        Search Strategy Used: {strategy}
        Content Quality: {confidence}

        Instructions:
        1. Provide actionable, specific advice
        2. Use examples when possible
        3. Keep the tone encouraging and supportive
        4. If information is insufficient, explain what you'd need to give a better answer

        Response:
        """)

        self.no_answer_response = "Sorry bro, I couldn't find enough info to answer that confidently."

        logger.info(" Building agent chain...")
        self._build_agent_chain()
        logger.info(" Rizzbot initialization complete!")

    def _load_env(self):
        from dotenv import load_dotenv, find_dotenv
        logger.info(" Loading environment variables...")
        return load_dotenv(find_dotenv())

    def _embed_question(self, question: str) -> List[float]:
        logger.info(f" Embedding question: '{question[:50]}...'")
        start_time = time.time()
        embedding = self.embeddings.embed_query(question)
        elapsed = time.time() - start_time
        logger.info(f" Embedding completed in {elapsed:.2f}s")
        return embedding

    def _calculate_confidence(self, docs: List, question: str) -> float:
        """Calculate confidence based on document relevance and count"""
        if not docs:
            return 0.0
        
        # Simple heuristic: more docs + longer content = higher confidence
        base_confidence = min(len(docs) / self.top_k, 1.0) * 0.7
        content_length_bonus = min(sum(len(str(doc)) for doc in docs) / 1000, 0.3)
        
        confidence = base_confidence + content_length_bonus
        logger.info(f" Calculated confidence: {confidence:.2f}")
        return confidence

    def _multi_query_search(self, question: str) -> tuple[List[str], float]:
        logger.info(f" Starting multi-query search for: '{question[:50]}...'")
        start_time = time.time()
        
        try:
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.full_vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
                llm=self.expand_llm
            )
            logger.info(" Generated query variations with MultiQueryRetriever")
            
            docs = retriever.get_relevant_documents(question)
            content = [doc.page_content for doc in docs]
            confidence = self._calculate_confidence(content, question)
            
            elapsed = time.time() - start_time
            logger.info(f" Multi-query search completed in {elapsed:.2f}s")
            logger.info(f" Retrieved {len(content)} documents")
            
            for i, doc in enumerate(content[:2]):  # Log first 2 docs
                logger.info(f" Doc {i+1} preview: {doc[:100]}...")
            
            return content, confidence
            
        except Exception as e:
            logger.error(f" Multi-query retrieval failed: {e}")
            return [], 0.0

    def _search_summaries(self, question: str) -> tuple[List[str], float]:
        logger.info(f" Searching summaries for: '{question[:50]}...'")
        start_time = time.time()
        
        try:
            docs = self.summaries_vectorstore.similarity_search(question, k=self.top_k)
            content = [doc.page_content for doc in docs]
            confidence = self._calculate_confidence(content, question)
            
            elapsed = time.time() - start_time
            logger.info(f" Summary search completed in {elapsed:.2f}s")
            logger.info(f" Retrieved {len(content)} summary documents")
            
            return content, confidence
            
        except Exception as e:
            logger.error(f" Summary search failed: {e}")
            return [], 0.0

    def _build_agent_chain(self):
        logger.info("Building agent execution chain...")
        
        # Simple approach - back to working version with enhancements
        def get_summaries(q):
            logger.info("Route: Using summaries search")
            content, confidence = self._search_summaries(q)
            logger.info(f"Summary search confidence: {confidence:.2f}")
            return content

        def get_full(q):
            logger.info("Route: Using full content search")
            content, confidence = self._multi_query_search(q)
            logger.info(f"Full search confidence: {confidence:.2f}")
            return content

        answer_from_summaries = (
            {
                "question": lambda q: q,
                "content": get_summaries,
                "confidence": lambda q: "High (Summary-based)",
                "strategy": lambda q: "Summary Search"
            }
            | self.enhanced_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        answer_from_full = (
            {
                "question": lambda q: q,
                "content": get_full,
                "confidence": lambda q: "Medium (Full-text)",
                "strategy": lambda q: "Multi-Query Full Search"
            }
            | self.enhanced_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        def has_summaries(q):
            logger.info("Checking if summaries are available...")
            summaries, confidence = self._search_summaries(q)
            has_good_summaries = len(summaries) > 0 and confidence > 0.3
            logger.info(f"Summary decision: {'YES' if has_good_summaries else 'NO'} (confidence: {confidence:.2f})")
            return has_good_summaries

        self.agent_chain = RunnableBranch(
            (has_summaries, answer_from_summaries),
            answer_from_full
        )
        logger.info("Agent chain built successfully")

    def answer_question(self, question: str) -> str:
        logger.info(f"\n{'='*60}")
        logger.info(f" RIZZBOT QUERY: {question}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            logger.info(" Invoking agent chain...")
            response = self.agent_chain.invoke(question)
            
            elapsed = time.time() - start_time
            logger.info(f" Query completed successfully in {elapsed:.2f}s")
            logger.info(f" Response length: {len(response)} characters")
            logger.info(f" Response preview: {response[:100]}...")
            
            return response
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f" Agentic pipeline failed after {elapsed:.2f}s: {e}")
            return self.no_answer_response

    def get_stats(self) -> Dict:
        """Get some basic stats about the system"""
        logger.info(" Generating system stats...")
        
        try:
            # Test connectivity
            summary_test = self.summaries_vectorstore.similarity_search("test", k=1)
            full_test = self.full_vectorstore.similarity_search("test", k=1)
            
            stats = {
                "summary_store_connected": len(summary_test) > 0,
                "full_store_connected": len(full_test) > 0,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "main_model": "gpt-4o",
                "expand_model": "gpt-3.5-turbo"
            }
            
            logger.info(f" System stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f" Failed to get stats: {e}")
            return {"error": str(e)}