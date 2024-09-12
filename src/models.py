from transformers import pipeline
from langchain.llms import CTransformers
from langchain.prompts import ChatPromptTemplate
import os

# Download models function
def download_models():
    """
    Downloads the necessary models for the chatbot using wget. 
    This function retrieves two models:
    1. BioMistral-7B - A medical-specific language model.
    2. Llama-2-7B - A general-purpose language model.
    """
    os.system('wget -O BioMistral-7B.Q4_K_M.gguf "https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf?download=true"')
    os.system('wget -O llama-2-7b.Q4_K_M.gguf "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf?download=true"')

# Call the download function to ensure the models are available before starting the bot
download_models()

# Text classification model for routing input
classification_labels = ['medical', 'general']

# Text classification model for routing input between medical or general models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load LLM models using CTransformers
general_llm = CTransformers(
    model="./llama-2-7b.Q4_K_M.gguf", 
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.7}
)

medical_llm = CTransformers(
    model="./BioMistral-7B.Q4_K_M.gguf", 
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.7}
)

# Prompt template for generating responses
template = """
You are a versatile AI assistant that can provide both medical advice and help users with general concerns, emotions, and questions outside the medical field. Your responses should be empathetic, supportive, and insightful, regardless of the topic.
Previous conversation history:
{history}
Current user input:
{input}
Guidelines for responding:
1. Provide a single, clear response directly addressing the user's current input.
2. Do not include any meta-information, instructions, or unrelated content in your response.
3. If the user mentions a medical concern, offer practical advice.
4. If the user talks about personal emotions, like feeling sad or hurt, respond with empathy and support.
5. Always keep your responses concise and relevant to the user's question or statement.
Your response:
"""

# Compile the prompt template using LangChain
prompt = ChatPromptTemplate.from_template(template)

# Function to route input to the correct model
def route_llm(user_input):
    """
    Routes user input to the appropriate LLM (medical or general).
    Parameters:
        user_input (str): The user's input message.
    Returns:
        CTransformers: The selected LLM model (general or medical).
    """
    result = classifier(user_input, classification_labels)
    label = result['labels'][0] if 'labels' in result else 'general'
    return medical_llm if label == 'medical' else general_llm
