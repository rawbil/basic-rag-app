from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Groq API key missing")

# if "GOOGLE_API_KEY" not in os.environ:
#     raise ValueError("GOOGLE API key missing")

groq_api_key = os.getenv("GROQ_API_KEY")
# google_api_key = os.getenv("GOOGLE_API_KEY")

# vector embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)


def ingest_documents(file_path):
    """
    Load document on server mount
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    reader = PdfReader(file_path)

    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages")

    docs = [
        Document(
            page_content=page.extract_text(),
            metadata={"source": str(file_path), "page": i + 1},
        )
        for i, page in enumerate(reader.pages)
    ]

    if not docs:
        raise ValueError("No readable text found in the PDF")

    # split docs

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Document could not be split into chunks")
    # Create vector DB for the first time
    Chroma.from_documents(
        embedding=embeddings, documents=chunks, persist_directory="chroma_db"
    )


def retrieve_documents():
    """retrieve document info on every request"""

    if not os.path.isdir("chroma_db"):
        raise FileNotFoundError(
            "Vector database not found. Please ingest documents first."
        )

    chroma_db = Chroma(embedding_function=embeddings, persist_directory="chroma_db")

    return chroma_db.as_retriever(search_kwargs={"k": 8})


async def GetRAGResponse(question):
    """
    Get results from pdf
    """
    try:
        retriever = retrieve_documents()

        # initialize model
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
        )

        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                   """
                    You are a RAG assistant. Answer based only on the provided context.

                    Write in plain, natural language without Markdown formatting.
                    Do not use Markdown syntax such as headings, bullets with asterisks, numbered Markdown lists, code blocks, tables, bold text, italics, or backticks.

                    If a list is useful, write it as simple sentences separated by line breaks, not Markdown bullets.

                    Give enough detail to be helpful and specific.
                    If the context does not contain enough information, clearly say what information is missing.

                    Context:
                    {context}
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        def format_documents(docs):
            if not docs:
                return "No relevant context found."

            return "\n\n".join([doc.page_content for doc in docs])

        # LCEL chain
        chain = (
            {"context": retriever | format_documents, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async for chunk in chain.astream(question):
            yield chunk

    except Exception as error:
        print(f"LLM Call failed: {error}")
        yield "\n\nSorry, the AI model failed when generating a response"
