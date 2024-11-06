from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_rag_chain(pdf_path, embedding_model, llm, chunk_size=400, chunk_overlap=50):
    print("inside createw rag chain")
    # Step 1: Load the PDF document (Syllabus)
    loader = PyPDFLoader("C:/Users/Minuli/PycharmProjects/MathChat/Algebraic_expressions.pdf")
    docs = loader.load()

    # Step 2: Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Split the documents into chunks
    splits = text_splitter.split_documents(docs)

    # Step 4: Create a vector store from the document chunks
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

    # Step 5: Create a retriever object to get data from the vector store
    retriever = vectorstore.as_retriever()

    # Step 6: Define the system prompt
    system_prompt = (
        "You are an intelligent chatbot. Use the following context to  retrieve the content related to the given "
        "problem from the syllabus\n\n{context}"
    )

    # Step 7: Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Step 8: Create the QA chain
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # Step 9: Create the RAG chain (QA chain + retriever)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Step 10: Return the RAG chain
    return rag_chain
