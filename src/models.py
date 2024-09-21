import os
import warnings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.llms import CTransformers
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

def download_models():
    """
    Downloads the necessary models for the chatbot using wget. 
    This function retrieves one model:
    1. BioMistral-7B - A medical-specific language model.
    """
    os.system("wget -O BioMistral-7B.Q8_0.gguf 'https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q8_0.gguf?download=true'")

def load_llm(model_path="./BioMistral-7B.Q8_0.gguf"):
    """
    Load the language model (LLM) for the chatbot using the provided model path.
    Returns an LLM instance configured for the medical chatbot.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        medical_llm (CTransformers): LLM instance for generating chatbot responses.
    """
    medical_llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={'max_new_tokens': 1024, 'temperature': 0.1},
        stream=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    return medical_llm

def load_embeddings():
    """
    Load the embedding model for sentence encoding.
    
    Returns:
        embed (SentenceTransformer): Embedding model for encoding sentences.
    """
    embed = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    return embed
