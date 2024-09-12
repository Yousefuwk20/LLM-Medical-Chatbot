import gradio as gr
from langchain.chains import ConversationChain
from .memory import EnhancedInMemoryHistory, get_by_session_id
from .models import route_llm, prompt

# Function to process input and generate a response
def process_input(user_input, session_id='1'):
    """
    Processes the user input and generates a response using the conversation chain.
    Parameters:
        user_input (str): The user's input message.
        session_id (str): The session ID for the chat (default is "1").
    Returns:
        str: The generated response from the chatbot.
    """
    memory = get_by_session_id(session_id)
    
    if user_input.lower() == 'exit':
        return "Exiting the chat session."

    llm = route_llm(user_input)

    conversation_chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        input_key='input',
        verbose=True
    )

    response = conversation_chain.run({"input": user_input})
    memory.save_context({'input': user_input}, response)
    
    return response

# Gradio interface function to handle input
def chatbot_interface(user_input, chat_history=None, session_id="1"):
    """
    Interface function for Gradio to handle input and output between the user and the chatbot.
    Parameters:
        user_input (str): The user's input message.
        session_id (str): The session ID for the chat (default is "1").
        chat_history (list): List of previous chat messages in the format [[user, bot], ...]
    Returns:
        list: Updated chat history including the new user and bot messages.
    """
    if chat_history is None:
        chat_history = []
    
    # Greeting at the start of the chat
    if user_input == "":
        bot_response = "Hi there! How can I help you today?"
    else:
        bot_response = process_input(user_input, session_id)
    
    # Add user input and bot response to chat history
    chat_history.append([user_input, bot_response])
    
    return chat_history

# Gradio launch
def launch_gradio_interface():
    gr.Interface(
        fn=chatbot_interface,
        inputs=[gr.Textbox(lines=7, label="Your input", placeholder="Type your message here...")],
        outputs=gr.Chatbot(label="Chat History"),
        title="AI Chatbot",
        live=False
    ).launch()
