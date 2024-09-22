from src.models import download_models, load_llm, load_embeddings
from src.pinecone_setup import setup_pinecone, populate_pinecone
from src.data_processing import load_pdf_files, split_text
from src.conversation import get_memory, create_chain
from src.vectorstore import setup_vectorstore
from langchain.chains import RetrievalQA

download_models()

medical_llm = load_llm()
embed = load_embeddings()

documents = load_pdf_files()
text_chunks = split_text(documents)

api_key = "f7fc405f-be0b-4604-8bbc-074a1b84f755"
index = setup_pinecone(api_key)
populate_pinecone(index, text_chunks, embed)

vectorstore = setup_vectorstore(index)

qa = RetrievalQA.from_chain_type(
    llm=medical_llm,
    chain_type="stuff", 
    retriever=vectorstore.as_retriever()
)

memory = get_memory(session_id="1", llm=medical_llm)
medical_chain = create_chain(medical_llm, memory)

def handle_user_input(user_input, session_id="1"):
    """
    Process user input and generate a response using LLM chain and Pinecone retrieval.
    
    Args:
        user_input (str): User's message.
        session_id (str): Session ID for conversation tracking.
        
    Returns:
        response (str): Generated response from the chatbot.
    """
    try:
        retrieval_response = qa.run(user_input)

        full_input = f"User Input: {user_input}\n\nRetrieved Information: {retrieval_response}"

        response = medical_chain.run({"history": "", "input": full_input})
        
        return response
    except Exception as e:
        print(f"Error during processing: {e}")
        return "Sorry, an error occurred while processing your input."

# usage
response = handle_user_input("I have headache and fever. What should i do?", session_id="1")
print(response)
