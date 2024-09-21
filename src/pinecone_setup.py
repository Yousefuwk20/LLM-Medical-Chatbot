import glob
import time
from tqdm.auto import tqdm
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec

def setup_pinecone(api_key):
    """
    Sets up a Pinecone index for vector storage.
    
    Args:
        api_key (str): API key for accessing Pinecone.
        
    Returns:
        index (Pinecone.Index): Initialized Pinecone index.
    """
    pc = Pinecone(api_key=api_key)
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    
    index_name = 'medbase'
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(index_name, dimension=768, metric='dotproduct', spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    index = pc.Index(index_name)
    return index

files = glob.glob("data/Cognitive Behavior therapy.pdf")

def populate_pinecone(index, text_chunks, embed, pdf_files=files):

    """
    Populates a Pinecone index with text chunks and metadata.
    
    Args:
        index (Pinecone.Index): Pinecone index to populate.
        text_chunks (list): List of text chunks to encode and store.
        embed (SentenceTransformer): Embedding model to use for encoding.
        pdf_files (list): List of PDF file paths for metadata.
    """
    
    batch_limit = 100
    texts = []
    metadatas = []

# Iterate over the text_chunks and prepare for upsertion to Pinecone
    for i, text_chunk in enumerate(tqdm(text_chunks)):

        metadata = {
        'chunk': i,
        'source': pdf_files[i % len(pdf_files)]  
    }
    
        texts.append(text_chunk)
        metadatas.append(metadata)
    
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.encode(texts)  
            index.upsert(vectors=zip(ids, embeds, metadatas))  
        
            texts = []
            metadatas = []

# Handle any remaining texts and metadata after the loop
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.encode(texts).tolist()
        index.upsert(vectors=list(zip(ids, embeds, metadatas)))
