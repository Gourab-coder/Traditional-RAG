import os
from dotenv import load_dotenv
from src1.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str="faiss_store", embedding_model: str="all-MiniLM-L6-v2"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else :
            self.vectorstore.load()
        
        model = "openai/gpt-oss-120b"
        self.llm = ChatGroq(model= model)
        print(f"[INFO] Groq LLM initialized: {self.llm}")

    def search_and_summarize(self, query: str, top_k: int=5) -> str :
        results = self.vectorstore.query(query)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        prompt = f"""
            Summarize the following context for the query: '{query}'
            context: {context}
            Answer:
        """
        response = self.llm.invoke([prompt])
        return response.content
    
