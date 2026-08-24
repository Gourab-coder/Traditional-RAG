from src1.data_loader import load_all_documents
from src1.embedding import EmbeddingPipeline
from src1.vectorstore import FaissVectorStore
from src1.search import RAGSearch

# Example usage
if __name__ == "__main__" :
    docs = load_all_documents("data")
    # chunks = EmbeddingPipeline().chunk_documents(docs)
    # chunkvectors = EmbeddingPipeline().embed_chunks(chunks)

    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    # store.load()
    # print(store.query("what is operating system", top_k=3))

    rag_search = RAGSearch()
    query = "what are the git commands?"
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("Summary:", summary)


    # print(chunkvectors)