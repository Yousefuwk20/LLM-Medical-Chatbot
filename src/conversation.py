from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

session_memory = {}

def get_memory(session_id="1", llm=None):
    """
    Retrieve or create a conversation memory buffer for a session.
    
    Args:
        session_id (str): Unique ID for the conversation session.
        llm (LLM): Language model for generating conversation summaries.
        
    Returns:
        memory (ConversationSummaryBufferMemory): Conversation memory object.
    """
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024)
    return session_memory[session_id]

def build_prompt_template():
    """
    Creates a prompt template for medical and general conversation responses.
    
    Returns:
        medical_prompt (PromptTemplate): Template for generating responses.
    """
    medical_template = """
    You are a versatile AI assistant that can provide both medical advice and help users with general concerns, emotions, and questions outside the medical field. Your responses should be empathetic, supportive, and insightful, regardless of the topic.
    Previous conversation history:
    {history}
    Current user input:
    {input}
    Guidelines for responding:
    1. Provide a single, clear response directly addressing the user's current input.
    2. Do not include any meta-information, instructions, or unrelated content in your response.
    3. If the user mentions a medical concern, offer practical advice.
    4. Always keep your responses concise and relevant to the user's question or statement.
    Your response:
    """
    
    medical_prompt = PromptTemplate(input_variables=["history", "input"], template=medical_template)
    return medical_prompt

def create_chain(llm, memory):
    """
    Creates a conversation chain using a language model and conversation memory.
    
    Args:
        llm (LLM): Language model for generating responses.
        memory (ConversationSummaryBufferMemory): Conversation memory for tracking history.
        
    Returns:
        medical_chain (LLMChain): Chain to handle conversations.
    """
    medical_prompt = build_prompt_template()
    medical_chain = LLMChain(llm=llm, prompt=medical_prompt, memory=memory)
    return medical_chain
