# rizzbot_agentic.py with logging

import os
import numpy as np
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


class Rizzbot:
    def __init__(self):
        print("[INIT] Starting Rizzbot initialization...")
        _ = self._load_env()
        self.similarity_threshold = 0.65
        self.top_k = 3

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

        for doc in docs:
            try:
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                sim = self._cosine_similarity(query_embedding, doc_embedding)
                print(f"[Similarity] Score: {sim:.4f} | Text: {doc.page_content[:80]}...")

                if sim >= threshold:
                    filtered.append(doc)
            except Exception as e:
                print(f"[Similarity] Failed to embed doc: {e}")

        return filtered

    def _hybrid_query_search(self, question: str) -> List[str]:
        print(f"[Search:Hybrid] Embedding question...")
        question_embedding = self._embed_question(question)
        combined_results = []

        for label, vectorstore in [("summaries", self.summaries_vectorstore), ("full", self.full_vectorstore)]:
            print(f"[Search:Hybrid] Trying {label} vectorstore...")

            try:
                retriever = MultiQueryRetriever.from_llm(
                    retriever=vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
                    llm=self.expand_llm
                )
                docs = retriever.invoke(question)
                filtered = self._filter_by_similarity(question_embedding, docs, self.similarity_threshold)
                print(f"[Search:Hybrid] {len(filtered)} docs passed threshold in {label}.")
                combined_results.extend([doc.page_content for doc in filtered])
            except Exception as e:
                print(f"[Search:Hybrid] Retrieval failed for {label}: {e}")

        return combined_results

    def _build_agent_chain(self):
        print("[Chain] Building agent chain...")

        def hybrid_search(q):
            results = self._hybrid_query_search(q)
            if results:
                return "\n\n".join(results)  # merge the filtered context into one string
            else:
                return self.no_answer_response
    
        self.agent_chain = (
            {
                "question": lambda q: q,
                "content": hybrid_search,
            }
            | self.base_prompt_template
            | self.main_llm
            | StrOutputParser()
        )

        print("[Chain] Agent chain constructed.")

    def answer_question(self, question: str) -> str:
        print(f"[Answer] Received question: {question}")
        try:
            context = self._hybrid_query_search(question)
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


if __name__ == "__main__":
    print("[Test] Starting test run for Rizzbot...\n")
    
    bot = Rizzbot()
    sample_question = "What is the capital of Russia?"

    print(f"\n[Test] Asking: {sample_question}\n")
    answer = bot.answer_question(sample_question)

    print("\n[Test] Final Answer:")
    print(answer)
