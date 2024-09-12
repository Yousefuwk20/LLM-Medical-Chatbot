# AI Chatbot Project

This project is an AI-powered chatbot that provides both medical advice and general assistance. It leverages two distinct language models to handle medical and general inquiries, ensuring versatile and empathetic responses.

"In this phase, we have successfully implemented the first feature from our proposal: 'Offering preliminary medical advice based on user input.' This was prioritized due to time constraints and the need to address latency issues."

## Prerequisites
- Python 3.10
- Gradio for the user interface
- Hugging Face Transformers and LangChain
- Required models: **BioMistral-7B** (medical) and **Llama-2-7B** (general-purpose)
- wget (for model downloading)

## Project Structure

The project is structured as follows:
``` css
root/ 
│ ├── app.py 
│ ├── requirements.txt 
│ └── README.md 
└── src/ 
     ├── init.py 
     ├── chatbot.py 
     ├── memory.py 
     └── models.py
```


### File Descriptions

- **app.py**: Main entry point to run the application, launching the Gradio interface.
- **requirements.txt**: Contains the list of dependencies required to run the chatbot.
- **README.md**: Documentation file providing a detailed overview of the project.
- **src/**:
  - **`__init__.py`**: Initializes the `src` package.
  - **chatbot.py**: Contains the chatbot logic and Gradio interface functions.
  - **memory.py**: Implements a custom memory class to manage chat history with timestamps.
  - **models.py**: Manages loading and routing between different language models.

## Setup Instructions

To run the project locally, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/https://github.com/TabibSystem/AI-Repo.git
    cd AI-Repo
    ```

2. **Install the Required Packages**: Make sure you have Python installed, then run:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Application**: Execute the following command to start the Gradio interface:
    ```sh
    python app.py
    ```

## Detailed Functionality

### Memory Management

The `EnhancedInMemoryHistory` class in `memory.py` is a custom memory class designed to manage conversation history. It includes:

- **Attributes**:
    - `messages`: Stores the exchanged messages.
    - `timestamps`: Records the timestamps of each message.

- **Methods**:
    - `add_messages`: Adds new messages to the memory with timestamps.
    - `get_recent_messages`: Retrieves the most recent messages.
    - `clear`: Clears the stored messages and timestamps.
    - `load_memory_variables`: Loads the memory variables for the conversation chain.
    - `save_context`: Saves the context of a conversation.
    - `clear_memory`: Alias for the clear function.

### Model Routing

The chatbot utilizes two different language models:

- **BioMistral-7B**: Specialized in handling medical-related queries.
- **Llama-2-7B**: A general-purpose model for non-medical questions.

The `route_llm` function in `models.py` uses a zero-shot classifier (`facebook/bart-large-mnli`) to determine whether a user's input is medical or general, routing the input accordingly.

### Chatbot Interface

The chatbot interface is built using Gradio:

- **Function**: `chatbot_interface` in `chatbot.py`
    - Manages input and output between the user and the chatbot.
    - Updates chat history and provides a real-time response.

- **Gradio Launch**:
    - The `launch_gradio_interface` function initializes and launches the web interface.

### Prompt Design

A prompt template is defined to guide the chatbot's responses. It includes instructions to provide empathetic, supportive, and context-aware replies, whether dealing with medical or general inquiries.

### Deployment
- The model has been deployed on Hugging Face for this phase of the project.
- We created a dedicated Space for the chatbot and utilized Gradio to handle user interactions and inferences.
- Due to budget constraints, we opted for the free resources plan available on Hugging Face.

### Cons
- **Latency in Model Performance**: The current setup experiences latency issues due to insufficient hardware and budget constraints.
- **Reduced Model Precision**: The chatbot operates with a quantized (4-bit) model instead of the more accurate 32-bit model, leading to potential reductions in response quality.
- **Lack of Arabic Language Support**: The chatbot currently does not support the Arabic language, as it hasn't been trained or fine-tuned on Arabic data.
    
### It's Future Improvements
- **Optimized GPU Utilization**: Transitioning to optimized GPUs in the cloud can significantly reduce latency and improve overall performance.
- **Full Model Precision**: Leveraging cloud resources will allow us to run the model at full 32-bit precision, enhancing response accuracy and effectiveness.
- **Arabic Language Support**: By integrating a model fine-tuned on Arabic data, we can expand the chatbot's capabilities to include support for Arabic language interactions.

