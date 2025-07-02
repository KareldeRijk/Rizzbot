# rizzbot_agentic.py

import os
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


# ... keep imports as before

class Rizzbot:
    def __init__(self):
        _ = self._load_env()
        self.similarity_threshold = 0.85
        self.top_k = 3

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "rizzbot"
        self.client = Client()

        self.main_llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
        self.expand_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.summaries_vectorstore = PineconeVectorStore(
            index_name="rizzbot-summaries-full-text",
            embedding=self.embeddings,
            text_key="full_text"
        )
        self.full_vectorstore = PineconeVectorStore(
            index_name="rizzbot", embedding=self.embeddings, text_key="full_text"
        )

        self.base_prompt_template = ChatPromptTemplate.from_template("""
        You are a charisma and personal development expert helping someone improve their social skills.

        Context: {content}
        Question: {question}

        Instructions:
        1. Provide actionable, specific advice
        2. Use examples when possible
        3. Keep the tone encouraging and supportive
        4. If information is insufficient, explain what you'd need to give a better answer

        Response:
        """)

        self.no_answer_response = "Sorry bro, I couldn't find enough info to answer that confidently."

        self._build_agent_chain()

    def _load_env(self):
        from dotenv import load_dotenv, find_dotenv
        return load_dotenv(find_dotenv())

    def _embed_question(self, question: str) -> List[float]:
        return self.embeddings.embed_query(question)

    def _multi_query_search(self, question: str) -> List[str]:
        try:
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.full_vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
                llm=self.expand_llm
            )
            docs = retriever.get_relevant_documents(question)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Multi-query retrieval failed: {e}")
            return []

    def _search_summaries(self, question: str) -> List[str]:
        try:
            docs = self.summaries_vectorstore.similarity_search(question, k=self.top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Summary search via vectorstore failed: {e}")
            return []

    def _build_agent_chain(self):
        get_summaries = lambda q: self._search_summaries(q)
        get_full = lambda q: self._multi_query_search(q)

        answer_from_summaries = (
            {
                "question": lambda q: q,
                "content": get_summaries,
            }
            | self.base_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        answer_from_full = (
            {
                "question": lambda q: q,
                "content": get_full,
            }
            | self.base_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        has_summaries = lambda q: len(self._search_summaries(q)) > 0

        self.agent_chain = RunnableBranch(
            (has_summaries, answer_from_summaries),
            answer_from_full
        )

    def answer_question(self, question: str) -> str:
        try:
            return self.agent_chain.invoke(question)
        except Exception as e:
            print(f"Agentic pipeline failed: {e}")
            return self.no_answer_response

