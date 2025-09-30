from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st
import chromadb.api

# Load environment variables from .env file
load_dotenv()

# Check if the key is available
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")

# Initialize OpenAI LLM and embeddings
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None
)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Chunking function
def chunking(pdf_data):
    print("Chunking PDF...")
    text_splitter = SemanticChunker(embeddings)
    docs = text_splitter.create_documents([pdf_data])
    return docs

# Function to fetch arXiv papers
def arxiv(topic):
    print("Retrieving research...")
    loader = ArxivLoader(
        query=topic,
        load_max_docs=2,
        load_all_available_meta=True,
    )
    docs = loader.load()
    ref_json = []
    result = []
    for doc in docs:
        temp_json = {
            'Published': doc.metadata['Published'], 
            'Title': doc.metadata['Title'],
            'Authors': doc.metadata['Authors'],
            'Summary': doc.metadata['Summary'],  
            'Page_content': doc.page_content
        }
        temp_json2 = {
            'Title': doc.metadata['Title'],
            'Summary': doc.metadata['Summary'],  
        }
        ref_json.append(temp_json)
        result.append(temp_json2)

    with open("arxiv_docs.json", "w", encoding="utf-8") as f:
        json.dump(ref_json, f, ensure_ascii=False, indent=4)

    return result

# Main RAG function
def call_rag(file_path=None, question="", include_arxis=False):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    arxiv_data = []
    retriever = None
    pdf_citation = None

    # Process the PDF if provided
    if file_path:
        print("Reading PDF...")
        loader = PyPDFLoader(file_path)
        document = loader.load()
        pdf_citation = file_path

        pdf_data = [{
            'Title': 'Uploaded PDF',
            'Authors': 'Unknown',
            'Summary': document[0].page_content[:500],
            'Page_content': document[0].page_content
        }]

        with open("pdf_docs.json", "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=4)

        chunks = chunking(document[0].page_content)
        vector_store = Chroma.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # Include arXiv data if enabled
    if include_arxis:
        arxiv_data = arxiv(question)

    # Set up QA chain
    if retriever or arxiv_data:
        print("In retrieval...")
        combined_content = f"ArXiv content found: {arxiv_data}" if arxiv_data else ""

        # Only create RetrievalQA if retriever exists
        if retriever:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,  # removed "if retriever else None"
                return_source_documents=True
            )

            query = f"""
            {combined_content}
            Question: {question}
            """
            result = qa_chain.invoke({"query": query})
            source_check = result["source_documents"][0].page_content if retriever else ""
            result_text = result['result']
        else:
            # If no retriever, only arXiv data
            result_text = "\n\n".join([f"{d['Title']}: {d['Summary']}" for d in arxiv_data])
            source_check = ""

        # Prepare citations
        citations = []
        if retriever:
            citations.append({"Source": pdf_citation})
        for data_source in arxiv_data:
            if data_source['Summary']:
                citations.append({
                    "Title": data_source.get("Title", ""),
                    "Published": data_source.get("Published", "")
                })

        return {
            "result": result_text,
            "citations": citations
        }
    else:
        return {
            "result": "Unable to process the request. Please provide a PDF or enable arXiv data.",
            "citations": []
        }


