import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_files(pdf_folder="data"):
    """
    Loads PDF files from a specified folder and extracts their content.

    Args:
        pdf_folder (str): Folder where PDF files are stored.
        
    Returns:
        documents (list): List of document objects extracted from PDFs.
    """
    pdf_files = glob.glob(f"{pdf_folder}/*.pdf")
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    return documents

def split_text(extracted_data, chunk_size=400, chunk_overlap=40):
    """
    Splits extracted text into smaller chunks for processing.
    
    Args:
        extracted_data (list): List of extracted document text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap size between chunks.
        
    Returns:
        text_list (list): List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(extracted_data)
    text_list = [chunk.page_content for chunk in text_chunks]
    return text_list
