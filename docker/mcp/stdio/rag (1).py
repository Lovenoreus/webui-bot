import os
import PyPDF2

from langchain_community.document_loaders import PyPDFLoader

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy.pool import QueuePool
from settings import SystemSettings
from urllib.parse import quote_plus

#import tiktoken
#from langchain.text_splitter import TokenTextSplitter
# Split text into chunks of 512 tokens, with 20% token overlap
#text_splitter = TokenTextSplitter(chunk_size=512,chunk_overlap=103)


settings = SystemSettings()
print(settings)

OLLAMA_BASE_URL = settings.ollama.direct_url
CONNECTION_STR = (
    f"postgresql+psycopg2://"
    f"{quote_plus(settings.postgres.pg_username)}:"
    f"{quote_plus(settings.postgres.pg_password)}@"
    f"{settings.postgres.pg_host}:"
    f"{settings.postgres.pg_port}/"
    f"{settings.postgres.pg_database}"
    #"?sslmode=verify-full" # Aktivera på servern
)

# 1. Create the model
llm = OllamaLLM(model=settings.ollama.llm_model,
                base_url=settings.ollama.direct_url,
                temperature=0.0)
embeddings = OllamaEmbeddings(model=settings.ollama.embedding_model, base_url=settings.ollama.direct_url)


def extract_text_from_pdf(path):
    try:
        reader = PyPDF2.PdfReader(path)
        pages = []
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_pdfs_as_documents(dir_path, limit=100, chunk_size=1000, overlap=200):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(".pdf")]
    files = sorted(files)[:limit]
    documents = []
    for idx, path in enumerate(files):
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
        for ci, chunk in enumerate(chunks):
            documents.append(Document(page_content=chunk,
                                      metadata={"source": path, "file_index": idx, "chunk_index": ci}))
    return documents



def get_vectorstorage(docs, embeddings, collection_name="documents_collection_mek1"):
    return PGVector.from_documents(
        documents= docs,
        embedding = embeddings,
        collection_name= collection_name,
        distance_strategy = DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)



def save_to_pgvector(documents, embeddings, collection_name="documents_collection_mek1"):
    # Example connection string, replace with your actual database credentials
    vectorestore = PGVector.from_documents(
        embedding=embeddings,
        documents=documents,
        collection_name=collection_name,
        connection_string=CONNECTION_STR,
        use_jsonb=True,
    )
    print("vectorstore :::: ")
    print(vectorestore)


def embedd_and_store_documents(pdf_dir, limit=100):
    docs = load_pdfs_as_documents(pdf_dir, limit=limit)
    save_to_pgvector(docs, embeddings)
    print("-----------  done  ---------------------")
    print("Number of docs: ", len(docs))
    #print(embeddings)


# Create Document Parsing Function to String
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrive_similar_documents(user_promt, top_k=3):
    vectorstore = PGVector(
        embedding_function = embeddings,
        collection_name= "documents_collection_mek1",
        distance_strategy = DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    #docs =  retriever.get_relevant_documents(user_promt)

    ###docs_txt = format_docs(docs)
    # Create the Prompt Template
    prompt_template = """Use the context provided to answer
    the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do 
    not know the answer to their question based on the context
    provided and that you are sorry. Answer in Swedish.

    context: {context}

    question: {query}

    answer: """

    # Create Prompt Instance from template
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    # Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Query the RAG Chain
    answer = rag_chain.invoke(user_promt)
    print("\nAnswer:: ", answer)

if __name__ == "__main__":
    embedd_and_store_documents(PATH_TO_PDF, limit=10)
    print("\n---- retrive_similar_documents::\n")
    #retrive_similar_documents(user_promt="Hur kan jag öppna Mediaundersökningen?", top_k=5)

