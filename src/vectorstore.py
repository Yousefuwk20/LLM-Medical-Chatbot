from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone

def setup_vectorstore(index):
    """
    Set up the vector store for retrieval using Pinecone and sentence embeddings.

    Args:
        index (Pinecone.Index): Initialized Pinecone index.
    Returns:
        vectorstore (LangchainPinecone): A Pinecone-based vector store for retrieval.
    """
    text_field = 'text'
    embed = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

    vectorstore = Pinecone(
        index, 
        embed.embed_query,
        text_key=text_field
    )
    return vectorstore
